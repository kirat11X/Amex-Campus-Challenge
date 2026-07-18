"""
ProfitLens RAG — chunking.

Every chunk is a plain dict:
    {chunk_id, source, kind, location, text}

kind ∈ {code, data_profile, framework, context, slide, user}
Chunks never split mid-sentence or mid-line-of-code.
"""
from __future__ import annotations

import hashlib
import re

TARGET_CHARS = 800
OVERLAP_CHARS = 120
MAX_CHARS = 1_200


def make_chunk(text: str, source: str, kind: str, location: str) -> dict:
    """Build a chunk dict with a content-addressed id."""
    text = text.strip()
    payload = f"{source}|{location}|{text}"
    return {
        "chunk_id": hashlib.sha1(payload.encode("utf-8")).hexdigest(),
        "source": source,
        "kind": kind,
        "location": location,
        "text": text,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Generic prose splitter
# ═══════════════════════════════════════════════════════════════════════════

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def split_prose(text: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split text into ~target-char pieces on paragraph, then sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]

    # Atomic units: paragraphs, further split if a single paragraph is oversized.
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= target:
            units.append(para)
            continue
        buf = ""
        for sent in _SENT_END.split(para):
            if buf and len(buf) + len(sent) + 1 > target:
                units.append(buf.strip())
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            units.append(buf.strip())

    pieces: list[str] = []
    buf = ""
    for unit in units:
        if buf and len(buf) + len(unit) + 2 > target:
            pieces.append(buf.strip())
            # Carry the tail of the previous piece for context continuity.
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n\n{unit}" if tail else unit
        else:
            buf = f"{buf}\n\n{unit}" if buf else unit
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


# ═══════════════════════════════════════════════════════════════════════════
#  Markdown (header-aware)
# ═══════════════════════════════════════════════════════════════════════════

_HEADER = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def chunk_markdown(text: str, source: str, kind: str = "context") -> list[dict]:
    """One chunk per ##/### section (header + body). Long bodies split at
    paragraph boundaries with the header repeated on every piece."""
    matches = list(_HEADER.finditer(text))
    if not matches:
        return [
            make_chunk(piece, source, kind, f"part {i + 1}")
            for i, piece in enumerate(split_prose(text))
        ]

    chunks: list[dict] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        for i, piece in enumerate(split_prose(preamble)):
            chunks.append(make_chunk(piece, source, kind, f"preamble {i + 1}"))

    for i, m in enumerate(matches):
        header = m.group(0).strip()
        title = m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if not body:
            continue
        pieces = split_prose(body, target=MAX_CHARS)
        for j, piece in enumerate(pieces):
            loc = title if len(pieces) == 1 else f"{title} ({j + 1}/{len(pieces)})"
            chunks.append(make_chunk(f"{header}\n\n{piece}", source, kind, loc))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  Python source (code-aware)
# ═══════════════════════════════════════════════════════════════════════════

# Keyword → section label, used to make code citations self-describing.
_SECTION_HINTS = [
    (re.compile(r"\bfillna\b|\bdropna\b|missing", re.I), "missing-value handling"),
    (re.compile(r"^W\s*=\s*dict\(", re.M), "W weights"),
    (re.compile(r"^FRAMEWORK\s*=", re.M), "framework dict"),
    (re.compile(r"\bprofit\s*=|\brank\(pct", re.I), "scoring"),
    (re.compile(r"eq_text|equation", re.I), "equation-string builder"),
    (re.compile(r"openpyxl|to_excel|wb\.save|shutil\.copy", re.I), "submission writer"),
    (re.compile(r"^\s*(import|from)\s", re.M), "imports/config"),
]


def _section_label(body: str) -> str | None:
    for pattern, label in _SECTION_HINTS:
        if pattern.search(body):
            return label
    return None


def _statement_starts(lines: list[str], start: int) -> list[int]:
    """Line indices where a new top-level statement begins.

    Tracks bracket depth and triple-quoted strings so continuation lines and
    multi-line literals are never treated as boundaries.
    """
    starts, depth, in_str, quote = [], 0, False, ""
    expecting = True
    for i in range(start, len(lines)):
        line = lines[i]
        if expecting and not in_str and depth == 0 and line.strip():
            starts.append(i)
            expecting = False
        j = 0
        while j < len(line):
            if in_str:
                if line.startswith(quote, j):
                    in_str, j = False, j + 3
                    continue
            else:
                if line.startswith(('"""', "'''"), j):
                    in_str, quote, j = True, line[j:j + 3], j + 3
                    continue
                ch = line[j]
                if ch == "#":
                    break
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth = max(0, depth - 1)
                elif ch in "\"'":
                    end = line.find(ch, j + 1)
                    j = len(line) if end == -1 else end
            j += 1
        if not in_str and depth == 0 and not line.rstrip().endswith("\\"):
            expecting = True
    return starts


_DICT_KEY = re.compile(r'^\s{1,}["\']([^"\']+)["\']\s*:')


def _split_dict_literal(lines: list[str], lo: int, hi: int, source: str,
                        name: str) -> list[dict] | None:
    """One chunk per top-level key of a large dict literal (e.g. FRAMEWORK)."""
    key_lines = [i for i in range(lo + 1, hi) if _DICT_KEY.match(lines[i])]
    if len(key_lines) < 2:
        return None
    chunks = []
    for n, k in enumerate(key_lines):
        end = key_lines[n + 1] if n + 1 < len(key_lines) else hi
        key = _DICT_KEY.match(lines[k]).group(1)
        chunks.append(make_chunk(
            f"{name} entry — {key}:\n" + "\n".join(lines[k:end]),
            source, "code", f"lines {k + 1}-{end} ({name}: {key})"))
    return chunks


def chunk_python(text: str, source: str) -> list[dict]:
    """Split a Python file on top-level statement boundaries, never mid-line.

    The module docstring is its own chunk; large dict literals (W, FRAMEWORK)
    are split one chunk per entry; everything else is merged up to the target
    size. Location metadata is the 1-indexed line range plus a section label.
    """
    lines = text.splitlines()
    chunks: list[dict] = []
    start = 0  # 0-indexed line cursor

    # Module docstring becomes its own chunk.
    stripped = text.lstrip()
    if stripped.startswith(('"""', "'''")):
        quote = stripped[:3]
        offset = len(text) - len(stripped)
        end_q = text.find(quote, offset + 3)
        if end_q != -1:
            doc_end_line = text.count("\n", 0, end_q + 3)
            chunks.append(make_chunk(
                "\n".join(lines[: doc_end_line + 1]), source, "code",
                f"lines 1-{doc_end_line + 1} (module docstring)"))
            start = doc_end_line + 1

    cuts = sorted(set(_statement_starts(lines, start) + [len(lines)]))
    if len(cuts) < 2:
        return [c for c in chunks if c["text"]]

    def _emit(lo: int, hi: int):
        body = "\n".join(lines[lo:hi]).strip("\n")
        if not body.strip():
            return
        label = _section_label(body)
        loc = f"lines {lo + 1}-{hi}" + (f" ({label})" if label else "")
        chunks.append(make_chunk(body, source, "code", loc))

    seg_start = cuts[0]
    for idx in range(1, len(cuts)):
        seg_end = cuts[idx]
        # A big dict literal (W, FRAMEWORK) gets one chunk per entry.
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:dict\(|\{)",
                     lines[seg_start])
        if m and seg_end - seg_start > 8:
            per_key = _split_dict_literal(lines, seg_start, seg_end, source, m.group(1))
            if per_key:
                chunks.extend(per_key)
                seg_start = seg_end
                continue
        body = "\n".join(lines[seg_start:seg_end])
        next_len = (len("\n".join(lines[seg_start:cuts[idx + 1]]))
                    if idx + 1 < len(cuts) else 10 ** 9)
        if len(body) >= TARGET_CHARS or next_len > MAX_CHARS or idx == len(cuts) - 1:
            _emit(seg_start, seg_end)
            seg_start = seg_end
    return [c for c in chunks if c["text"]]


# ═══════════════════════════════════════════════════════════════════════════
#  Plain text / JSON
# ═══════════════════════════════════════════════════════════════════════════

def chunk_plaintext(text: str, source: str, kind: str = "user") -> list[dict]:
    return [
        make_chunk(piece, source, kind, f"part {i + 1}")
        for i, piece in enumerate(split_prose(text))
    ]
