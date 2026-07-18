"""
ProfitLens — Gemini LLM Integration
Loads PROFITLENS_CONTEXT.md verbatim as the system instruction.
Function-calling loop, retry/backoff, and history trimming are unchanged.
"""
from __future__ import annotations
import functools, logging, time
import streamlit as st
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL, CONTEXT_MD_PATH
from tools import TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6
MAX_HISTORY_MESSAGES = 12
RETRY_DELAYS = [2, 4, 8]

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
#  Gemini client
# ═══════════════════════════════════════════════════════════════════════════

def get_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY / API_KEY not set in .env")
    return genai.Client(api_key=GEMINI_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════
#  Tool declarations (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

_TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_customer_profile",
        description="Get full profile for a customer: raw features, P&L terms, profit, rank, top-20% flag.",
        parameters={"type": "OBJECT", "properties": {
            "customer_id": {"type": "INTEGER", "description": "Customer ID"}
        }, "required": ["customer_id"]}),
    types.FunctionDeclaration(
        name="get_top_customers",
        description="Get top/bottom N customers by profit with key drivers.",
        parameters={"type": "OBJECT", "properties": {
            "n": {"type": "INTEGER", "description": "Count (max 50, default 10)"},
            "ascending": {"type": "BOOLEAN", "description": "True for bottom N"}
        }}),
    types.FunctionDeclaration(
        name="segment_compare",
        description="Compare a feature between top-20% and rest: mean, median, p25, p75, missing%.",
        parameters={"type": "OBJECT", "properties": {
            "feature": {"type": "STRING", "description": "f1-f23 or term column name"}
        }, "required": ["feature"]}),
    types.FunctionDeclaration(
        name="term_contribution_summary",
        description="Aggregate mean of each P&L term for top-20% vs rest, showing biggest separators.",
        parameters={"type": "OBJECT", "properties": {}}),
    types.FunctionDeclaration(
        name="get_distribution",
        description="Histogram data for any column. App renders as plotly chart.",
        parameters={"type": "OBJECT", "properties": {
            "column": {"type": "STRING", "description": "Column name"},
            "bins": {"type": "INTEGER", "description": "Bins (default 40)"},
            "clip_percentiles": {"type": "ARRAY", "items": {"type": "NUMBER"},
                                 "description": "[lo,hi] percentiles"},
        }, "required": ["column"]}),
    types.FunctionDeclaration(
        name="query_segment",
        description="Guarded pandas query with aggregation for ad-hoc questions.",
        parameters={"type": "OBJECT", "properties": {
            "filters": {"type": "STRING", "description": "pandas query expression"},
            "agg": {"type": "STRING", "description": "count|mean_profit|sum_profit|top20_share"}
        }, "required": ["filters", "agg"]}),
    types.FunctionDeclaration(
        name="explain_formula",
        description="Return weights, equation, term table, and submission methodology.",
        parameters={"type": "OBJECT", "properties": {}}),
    types.FunctionDeclaration(
        name="search_knowledge_base",
        description=(
            "Semantic search over the ProfitLens knowledge base: the project's own "
            "code, methodology write-up, challenge deck, feature dictionary and "
            "generated data profiles ('core'), plus any files the user uploaded "
            "('user'). Returns text chunks with source and location for citation. "
            "Not a source of customer-level numbers or live aggregates."),
        parameters={"type": "OBJECT", "properties": {
            "query": {"type": "STRING", "description": "Natural-language search query"},
            "corpus": {"type": "STRING",
                       "description": "core | user | both (default both)"},
            "k": {"type": "INTEGER", "description": "Number of chunks (default 5, max 10)"},
        }, "required": ["query"]}),
]

# Appended to the system instruction at runtime — PROFITLENS_CONTEXT.md itself
# is never edited.
RAG_GUIDANCE = (
    "\n\n## KNOWLEDGE-BASE RETRIEVAL\n"
    "Use search_knowledge_base for: questions about the code/implementation, the "
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

GEMINI_TOOLS = [types.Tool(function_declarations=_TOOL_DECLARATIONS)]


# ═══════════════════════════════════════════════════════════════════════════
#  Tool execution (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def _execute_tool_calls(function_calls):
    responses = []
    for fc in function_calls:
        fn = TOOL_FUNCTIONS.get(fc.name)
        args = dict(fc.args) if fc.args else {}
        if fn is None:
            result = {"error": f"Unknown tool: {fc.name}"}
        else:
            try:
                result = fn(**args)
            except Exception as e:
                result = {"error": str(e)}
        responses.append(types.Part.from_function_response(name=fc.name, response=result))
    return responses


# ═══════════════════════════════════════════════════════════════════════════
#  Main chat function (runtime_block replaces stats_pack)
# ═══════════════════════════════════════════════════════════════════════════

def chat_with_gemini(user_message: str, history: list[dict], runtime_block: str = "",
                      free_tier: bool = True) -> tuple[str, list[dict], list[str]]:
    """
    Send user message through Gemini.

    Args:
        user_message:  The new user message.
        history:       List of {"role": "user"|"model", "text": str} dicts.
        runtime_block: Pre-built runtime verification text appended to system instruction.
        free_tier:     If True, skip the tool-calling loop entirely and make a single
                        direct generate_content call (1 request/message, safe for the
                        20-req/day free tier). If False, run the full multi-round
                        function-calling pipeline (up to MAX_TOOL_ROUNDS requests/message).

    Returns:
        (assistant_reply_text, updated_history, tool_names_called)
    """
    client = get_client()
    system_instruction, _ = load_context(runtime_block)

    contents = []
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    for msg in trimmed:
        contents.append(types.Content(role=msg["role"],
                                      parts=[types.Part.from_text(text=msg["text"])]))
    contents.append(types.Content(role="user",
                                  parts=[types.Part.from_text(text=user_message)]))

    if free_tier:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        )
        max_rounds = 1
    else:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=GEMINI_TOOLS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        )
        max_rounds = MAX_TOOL_ROUNDS

    tool_names_called = []
    response = None

    for _ in range(max_rounds):
        for delay in RETRY_DELAYS + [None]:
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL, contents=contents, config=config)
                break
            except Exception as e:
                err = str(e).lower()
                if ("429" in err or "rate" in err or "resource" in err) and delay:
                    st.toast(f"⏳ Rate limit, retrying in {delay}s…")
                    time.sleep(delay)
                else:
                    raise

        if response is None:
            break

        fc_list = response.function_calls
        if fc_list:
            for fc in fc_list:
                tool_names_called.append(fc.name)
            contents.append(response.candidates[0].content)
            fn_responses = _execute_tool_calls(fc_list)
            contents.append(types.Content(role="user", parts=fn_responses))
        else:
            text = response.text or "(No response)"
            new_hist = history + [{"role": "user", "text": user_message},
                                   {"role": "model", "text": text}]
            return text, new_hist, tool_names_called

    text = (response.text if response and response.text
            else "Ran out of tool rounds. Try a more specific question.")
    new_hist = history + [{"role": "user", "text": user_message},
                           {"role": "model", "text": text}]
    return text, new_hist, tool_names_called
