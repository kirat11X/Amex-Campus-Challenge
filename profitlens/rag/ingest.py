"""
ProfitLens RAG — ingestion.

Two corpora:
  core  — built from the project's own files (code, feature dictionary, dataset
          and formula-output profile docs, framework sheet, context file, deck).
          Rebuilt only when a source file's sha256 changes.
  user  — files the user uploads through the "Digest your own files?" tab.

Both live in rag_store/ and survive restarts.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from datetime import datetime

import pandas as pd

from config import (CHALLENGE_PDF, CONTEXT_MD_PATH, FEATURE_CSV, RAG_STORE_DIR,
                    SUBMISSION_XLSX, TRIAL_PY, W)
from rag.chunker import (chunk_markdown, chunk_plaintext, chunk_python,
                         make_chunk, split_prose)
from rag.embed import cache_stats, embed_texts
from rag.profile_docs import dataset_profile_docs, formula_output_docs, tabular_profile_docs
from rag.store import VectorStore

logger = logging.getLogger(__name__)

HASHES_PATH = RAG_STORE_DIR / "hashes.json"
USER_MANIFEST_PATH = RAG_STORE_DIR / "user_files.json"

# Bump whenever the chunking or profile-doc logic changes, so an existing
# core index is rebuilt even though the source files are untouched.
CORPUS_VERSION = 1

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTS = ["pdf", "docx", "txt", "md", "csv", "xlsx", "py", "json"]

_core = VectorStore("core")
_user = VectorStore("user")
_user_loaded = False


def core_store() -> VectorStore:
    return _core


def user_store() -> VectorStore:
    global _user_loaded
    if not _user_loaded:
        _user.load()
        _user_loaded = True
    return _user


# ═══════════════════════════════════════════════════════════════════════════
#  Hashing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sha256_file(path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _current_hashes() -> dict:
    sources = {
        "trial.py": TRIAL_PY,
        "feature_csv": FEATURE_CSV,
        "submission_xlsx": SUBMISSION_XLSX,
        "context_md": CONTEXT_MD_PATH,
        "challenge_pdf": CHALLENGE_PDF,
    }
    hashes = {k: _sha256_file(p) for k, p in sources.items()}
    # A weight change alters every generated profile doc, so fold W in too.
    hashes["weights"] = hashlib.sha256(
        json.dumps(W, sort_keys=True).encode()).hexdigest()
    hashes["corpus_version"] = CORPUS_VERSION
    return hashes


def _load_hashes() -> dict:
    if HASHES_PATH.exists():
        try:
            return json.loads(HASHES_PATH.read_text())
        except Exception:
            return {}
    return {}


# ═══════════════════════════════════════════════════════════════════════════
#  Core corpus
# ═══════════════════════════════════════════════════════════════════════════

def _pdf_pages(data: bytes) -> list[tuple[int, str]]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append((i, text))
    return pages


def _feature_dictionary_chunks() -> list[dict]:
    """One chunk per feature, merging description, formula role and usage."""
    from config import FEATURE_DICT
    from rag.profile_docs import _FEATURE_ROLE

    if not FEATURE_CSV.exists():
        return []
    fd = pd.read_csv(FEATURE_CSV)
    fd.columns = [c.strip() for c in fd.columns]
    name_col, desc_col = fd.columns[0], fd.columns[1]
    chunks = []
    for _, row in fd.iterrows():
        feat = str(row[name_col]).strip()
        if not feat.startswith("f"):
            continue
        desc = str(row[desc_col]).strip()
        role = _FEATURE_ROLE.get(feat)
        chunks.append(make_chunk(
            f"{feat} — {FEATURE_DICT.get(feat, desc)}.\n"
            f"Dataset description: {desc}.\n"
            f"Role in the profitability formula: {role or 'not used'}.\n"
            f"Used in formula: {'yes' if role else 'no'}.",
            "6a3eb1af6b994_feature_description.csv", "framework", f"feature {feat}"))
    return chunks


def _framework_sheet_chunks() -> list[dict]:
    if not SUBMISSION_XLSX.exists():
        return []
    try:
        fw = pd.read_excel(SUBMISSION_XLSX, sheet_name="Profitability Framework")
    except Exception as e:
        logger.warning("Could not read Profitability Framework sheet: %s", e)
        return []
    fw.columns = [str(c).strip() for c in fw.columns]
    sec_col, resp_col = fw.columns[0], fw.columns[1]
    chunks = []
    for _, row in fw.iterrows():
        section = str(row[sec_col]).strip()
        response = str(row[resp_col]).strip()
        if not section or section.lower() == "nan" or response.lower() == "nan":
            continue
        for i, piece in enumerate(split_prose(response, target=1_200)):
            loc = f"Profitability Framework / {section}"
            chunks.append(make_chunk(
                f"Profitability Framework — {section}:\n{piece}",
                "submission_FINAL.xlsx", "framework",
                loc if i == 0 else f"{loc} ({i + 1})"))
    return chunks


def build_core(df: pd.DataFrame, verification: dict | None = None,
               force: bool = False) -> dict:
    """Build (or load) the core index. Returns a status dict."""
    RAG_STORE_DIR.mkdir(parents=True, exist_ok=True)
    current, previous = _current_hashes(), _load_hashes()

    if not force and current == previous and _core.load():
        print(f"[ProfitLens RAG] Core index loaded from disk: "
              f"{len(_core):,} chunks (no re-embedding).")
        return {"rebuilt": False, "chunks": len(_core), "sources": _core.sources}

    chunks: list[dict] = []

    if TRIAL_PY.exists():
        chunks += chunk_python(TRIAL_PY.read_text(encoding="utf-8"), "trial.py")
    chunks += _feature_dictionary_chunks()
    chunks += dataset_profile_docs(df)
    chunks += formula_output_docs(df, verification)
    chunks += _framework_sheet_chunks()
    if CONTEXT_MD_PATH.exists():
        chunks += chunk_markdown(CONTEXT_MD_PATH.read_text(encoding="utf-8"),
                                 "PROFITLENS_CONTEXT.md", kind="context")
    if CHALLENGE_PDF.exists():
        try:
            for page_no, text in _pdf_pages(CHALLENGE_PDF.read_bytes()):
                for i, piece in enumerate(split_prose(text, target=1_200)):
                    chunks.append(make_chunk(
                        piece, CHALLENGE_PDF.name, "slide",
                        f"p.{page_no}" if i == 0 else f"p.{page_no} ({i + 1})"))
        except Exception as e:
            logger.warning("Challenge PDF could not be parsed: %s", e)

    # Deduplicate by chunk_id before paying for embeddings.
    seen, unique = set(), []
    for c in chunks:
        if c["chunk_id"] not in seen and c["text"]:
            seen.add(c["chunk_id"])
            unique.append(c)

    _core.clear()
    vectors = embed_texts([c["text"] for c in unique], "RETRIEVAL_DOCUMENT")
    _core.add(vectors, unique)
    _core.save()
    HASHES_PATH.write_text(json.dumps(current, indent=2))

    stats = cache_stats()
    print(f"[ProfitLens RAG] Core index built: {len(_core):,} chunks "
          f"(embedding cache: {stats['hits']} hits / {stats['misses']} misses).")
    return {"rebuilt": True, "chunks": len(_core), "sources": _core.sources}


# ═══════════════════════════════════════════════════════════════════════════
#  User corpus — manifest
# ═══════════════════════════════════════════════════════════════════════════

def _load_manifest() -> list[dict]:
    if USER_MANIFEST_PATH.exists():
        try:
            return json.loads(USER_MANIFEST_PATH.read_text())
        except Exception:
            return []
    return []


def _save_manifest(entries: list[dict]):
    RAG_STORE_DIR.mkdir(parents=True, exist_ok=True)
    USER_MANIFEST_PATH.write_text(json.dumps(entries, indent=2))


def list_user_files() -> list[dict]:
    """Manifest rows: filename, type, chunks, ingested_at, content_hash."""
    return _load_manifest()


# ═══════════════════════════════════════════════════════════════════════════
#  User corpus — parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse_upload(name: str, data: bytes) -> list[dict]:
    """Turn an uploaded file into chunks. Raises ValueError if unparseable."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type '.{ext}'. "
                         f"Supported: {', '.join('.' + e for e in SUPPORTED_EXTS)}")

    if ext == "pdf":
        pages = _pdf_pages(data)
        if not pages:
            raise ValueError("No extractable text (scanned or image-only PDF?)")
        chunks = []
        for page_no, text in pages:
            for i, piece in enumerate(split_prose(text, target=1_200)):
                chunks.append(make_chunk(
                    piece, name, "user",
                    f"p.{page_no}" if i == 0 else f"p.{page_no} ({i + 1})"))
        return chunks

    if ext == "docx":
        import docx
        doc = docx.Document(io.BytesIO(data))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            raise ValueError("Document contains no text paragraphs")
        return [make_chunk(piece, name, "user", f"part {i + 1}")
                for i, piece in enumerate(split_prose("\n\n".join(paragraphs)))]

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(data))
        return tabular_profile_docs(df, name, name)

    if ext == "xlsx":
        xls = pd.ExcelFile(io.BytesIO(data))
        chunks = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            chunks += tabular_profile_docs(df, name, f"{name} / {sheet}")
        if not chunks:
            raise ValueError("Workbook contains no readable sheets")
        return chunks

    if ext == "py":
        return chunk_python(data.decode("utf-8", errors="replace"), name)

    # txt / md / json
    text = data.decode("utf-8", errors="replace")
    if ext == "md":
        return chunk_markdown(text, name, kind="user")
    if ext == "json":
        try:
            text = json.dumps(json.loads(text), indent=2)
        except Exception:
            pass  # not valid JSON — index it as plain text
    return chunk_plaintext(text, name, kind="user")


# ═══════════════════════════════════════════════════════════════════════════
#  User corpus — mutation
# ═══════════════════════════════════════════════════════════════════════════

def ingest_user_file(name: str, data: bytes) -> dict:
    """Chunk, embed and add one uploaded file. Never raises on user error —
    returns {"ok": False, "message": ...} instead."""
    if len(data) > MAX_UPLOAD_BYTES:
        return {"ok": False, "message":
                f"{name} is {len(data) / 1e6:.1f} MB — the limit is 25 MB."}

    store = user_store()
    content_hash = hashlib.sha256(data).hexdigest()
    manifest = _load_manifest()
    if any(e["content_hash"] == content_hash for e in manifest):
        return {"ok": True, "duplicate": True, "chunks": 0,
                "message": f"{name} is already digested — skipped re-embedding."}

    try:
        chunks = _parse_upload(name, data)
    except Exception as e:
        return {"ok": False, "message": f"Could not read {name}: {e}"}
    if not chunks:
        return {"ok": False, "message": f"{name} contained no indexable text."}

    try:
        vectors = embed_texts([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    except Exception as e:
        # Embed BEFORE replacing, so a failed re-upload leaves the previously
        # digested version of the file intact.
        return {"ok": False, "message":
                f"Embedding failed for {name} (check your connection): {e}"}

    # A re-upload under the same name replaces the previous version.
    store.delete_by_source(name)
    manifest = [e for e in manifest if e["filename"] != name]

    added = store.add(vectors, chunks)
    store.save()

    manifest.append({
        "filename": name,
        "type": name.rsplit(".", 1)[-1].lower(),
        "chunks": added,
        "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "content_hash": content_hash,
    })
    _save_manifest(manifest)
    return {"ok": True, "duplicate": False, "chunks": added,
            "message": f"✓ {name} digested — {added} chunks"}


def delete_user_file(name: str) -> int:
    """Remove one file's vectors (rebuilds the small user index). Returns count."""
    store = user_store()
    removed = store.delete_by_source(name)
    store.save()
    _save_manifest([e for e in _load_manifest() if e["filename"] != name])
    return removed


def clear_user_corpus() -> int:
    store = user_store()
    removed = len(store)
    store.clear()
    store.save()
    _save_manifest([])
    return removed
