"""
ProfitLens — local LLM integration (Ollama)
Loads PROFITLENS_CONTEXT.md verbatim as the system instruction.
Function-calling loop and history trimming run against a local Ollama model.
"""
from __future__ import annotations
import functools, json, logging
import requests
from config import OLLAMA_HOST, LLM_MODEL, CONTEXT_MD_PATH
from tools import TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6
MAX_HISTORY_MESSAGES = 12
NUM_CTX = 8192                # system context alone is ~5k tokens; keeps the model fully on a 6 GB GPU
MAX_TOOL_RESULT_CHARS = 4000  # keep tool output inside the small model's window
REQUEST_TIMEOUT = 600

_FALLBACK_SYSTEM_PROMPT = (
    "You are ProfitLens, an issuer-side profitability analyst for the Amex Premier Card.\n"
    "Ground every number in tool outputs only — never estimate or invent values.\n"
    "Use the available tools to answer questions about customer profitability.\n"
    "The profitability formula and its weights are frozen — explain them, do not change them.\n"
    "Currency is USD. Missing feature values mean no recorded activity, scored as 0.\n"
)


# ═══════════════════════════════════════════════════════════════════════════
#  Context loader (reads once per process — never paraphrased or truncated)
# ═══════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=1)
def _read_context_file() -> str | None:
    """Read PROFITLENS_CONTEXT.md verbatim. Returns None if file is missing."""
    if not CONTEXT_MD_PATH.exists():
        return None
    with open(CONTEXT_MD_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def load_context(runtime_block: str = "") -> tuple[str, bool]:
    """
    Load the grounding context file and append the runtime verification block.

    Returns:
        (system_instruction_text, context_file_found: bool)
    """
    raw = _read_context_file()
    if raw is None:
        logger.error("PROFITLENS_CONTEXT.md not found at %s — using fallback prompt.", CONTEXT_MD_PATH)
        return _FALLBACK_SYSTEM_PROMPT + runtime_block + RAG_GUIDANCE, False
    text = raw + runtime_block + RAG_GUIDANCE
    # Token estimate (rough: chars / 4)
    token_est = len(text) // 4
    print(f"[ProfitLens] System instruction: ~{token_est} tokens "
          f"({'context file' if raw else 'FALLBACK'} + runtime block).")
    return text, True


# ═══════════════════════════════════════════════════════════════════════════
#  Ollama client
# ═══════════════════════════════════════════════════════════════════════════

def check_ollama() -> str | None:
    """Return None if Ollama is up and LLM_MODEL is pulled, else an error message."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException:
        return f"Ollama is not reachable at {OLLAMA_HOST}. Start it with `ollama serve`."
    names = {m["name"] for m in r.json().get("models", [])}
    if LLM_MODEL not in names and f"{LLM_MODEL}:latest" not in names:
        return f"Model `{LLM_MODEL}` is not installed. Run `ollama pull {LLM_MODEL}`."
    return None


def _fn(name: str, description: str, properties: dict | None = None,
        required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": required or []}}}


# ═══════════════════════════════════════════════════════════════════════════
#  Tool declarations
# ═══════════════════════════════════════════════════════════════════════════

TOOLS = [
    _fn("get_customer_profile",
        "Get full profile for a customer: raw features, P&L terms, profit, rank, top-20% flag.",
        {"customer_id": {"type": "integer", "description": "Customer ID"}},
        ["customer_id"]),
    _fn("get_top_customers",
        "Get top/bottom N customers by profit with key drivers.",
        {"n": {"type": "integer", "description": "Count (max 50, default 10)"},
         "ascending": {"type": "boolean", "description": "True for bottom N"}}),
    _fn("segment_compare",
        "Compare a feature between top-20% and rest: mean, median, p25, p75, missing%.",
        {"feature": {"type": "string", "description": "f1-f23 or term column name"}},
        ["feature"]),
    _fn("term_contribution_summary",
        "Aggregate mean of each P&L term for top-20% vs rest, showing biggest separators."),
    _fn("get_distribution",
        "Histogram data for any column. App renders as plotly chart.",
        {"column": {"type": "string", "description": "Column name"},
         "bins": {"type": "integer", "description": "Bins (default 40)"},
         "clip_percentiles": {"type": "array", "items": {"type": "number"},
                              "description": "[lo,hi] percentiles"}},
        ["column"]),
    _fn("query_segment",
        "Guarded pandas query with aggregation for ad-hoc questions.",
        {"filters": {"type": "string", "description": "pandas query expression"},
         "agg": {"type": "string",
                 "description": "count|mean_profit|sum_profit|top20_share"}},
        ["filters", "agg"]),
    _fn("explain_formula",
        "Return weights, equation, term table, and submission methodology."),
    _fn("search_knowledge_base",
        "Semantic search over the ProfitLens knowledge base: the project's own "
        "code, methodology write-up, challenge deck, feature dictionary and "
        "generated data profiles ('core'), plus any files the user uploaded "
        "('user'). Returns text chunks with source and location for citation. "
        "Not a source of customer-level numbers or live aggregates.",
        {"query": {"type": "string", "description": "Natural-language search query"},
         "corpus": {"type": "string", "description": "core | user | both (default both)"},
         "k": {"type": "integer", "description": "Number of chunks (default 5, max 10)"}},
        ["query"]),
]

# Appended to the system instruction at runtime — PROFITLENS_CONTEXT.md itself
# is never edited.
RAG_GUIDANCE = (
    "\n\n## KNOWLEDGE-BASE RETRIEVAL\n"
    "You MUST call search_knowledge_base BEFORE answering — never say something "
    "is unavailable without searching first — for: questions about the code/implementation, the "
    "methodology write-up, the challenge rules/deck, feature meanings, and ANY "
    "question about user-uploaded files. Do not use it for customer-level numbers "
    "or live aggregates — those come from the pandas tools. When you answer from "
    "retrieved chunks, cite the source and location (e.g. 'per trial.py lines 40-55' "
    "or 'from your uploaded report.pdf, p.3'). If retrieval returns nothing "
    "relevant, say so.\n"
    "Trust order: live pandas tool result > Section 8 stats > retrieved chunk "
    "(cite it) > refuse. For user-uploaded files, retrieved chunks ARE the primary "
    "source.\n"
)


# ═══════════════════════════════════════════════════════════════════════════
#  Tool execution
# ═══════════════════════════════════════════════════════════════════════════

def _execute_tool_call(name: str, args: dict) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        result = {"error": f"Unknown tool: {name}"}
    else:
        try:
            result = fn(**args)
        except Exception as e:
            result = {"error": str(e)}
    text = json.dumps(result, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + '... [truncated]"}'
    return text


def _prefetch_context(query: str, k: int = 4) -> str:
    """Always-on retrieval for tool mode. A 4B model won't reliably choose to call
    search_knowledge_base, so the top chunks are injected up front."""
    try:
        hits = TOOL_FUNCTIONS["search_knowledge_base"](query, "both", k).get("results", [])
    except Exception as e:
        logger.warning("Prefetch retrieval failed: %s", e)
        return ""
    if not hits:
        return ""
    blocks = [f"[{h['source']} — {h['location']}]\n{h['text'][:1200]}" for h in hits]
    return ("\n\n---\nRetrieved knowledge-base excerpts (cite source + location "
            "when you use them):\n\n" + "\n\n".join(blocks))


def _chat(messages: list[dict], tools: list[dict] | None) -> dict:
    payload = {
        "model": LLM_MODEL, "messages": messages, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": NUM_CTX},
    }
    if tools:
        payload["tools"] = tools
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload,
                             timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
    return resp.json()["message"]


# ═══════════════════════════════════════════════════════════════════════════
#  Main chat function (runtime_block replaces stats_pack)
# ═══════════════════════════════════════════════════════════════════════════

def chat_with_llm(user_message: str, history: list[dict], runtime_block: str = "",
                  use_tools: bool = True) -> tuple[str, list[dict], list[str]]:
    """
    Send user message through the local Ollama model.

    Args:
        user_message:  The new user message.
        history:       List of {"role": "user"|"model", "text": str} dicts.
        runtime_block: Pre-built runtime verification text appended to system instruction.
        use_tools:     If True, run the multi-round function-calling pipeline (pandas
                       tools + knowledge-base retrieval, up to MAX_TOOL_ROUNDS model
                       calls). If False, answer directly from the system context.

    Returns:
        (assistant_reply_text, updated_history, tool_names_called)
    """
    system_instruction, _ = load_context(runtime_block)

    messages = [{"role": "system", "content": system_instruction}]
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        messages.append({"role": "assistant" if msg["role"] == "model" else "user",
                         "content": msg["text"]})
    extra = _prefetch_context(user_message) if use_tools else ""
    messages.append({"role": "user", "content": user_message + extra})

    tools = TOOLS if use_tools else None
    tool_names_called: list[str] = []
    text = None

    for _ in range(MAX_TOOL_ROUNDS if use_tools else 1):
        reply = _chat(messages, tools)
        calls = reply.get("tool_calls") or []
        if not calls:
            text = (reply.get("content") or "").strip() or "(No response)"
            break
        messages.append(reply)
        for call in calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            tool_names_called.append(name)
            messages.append({"role": "tool", "tool_name": name,
                             "content": _execute_tool_call(name, args)})

    if text is None:
        text = "Ran out of tool rounds. Try a more specific question."
    new_hist = history + [{"role": "user", "text": user_message},
                          {"role": "model", "text": text}]
    return text, new_hist, tool_names_called
