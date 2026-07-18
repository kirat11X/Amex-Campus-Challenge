"""
ProfitLens — Streamlit Chat UI
Premier Card Profitability Chatbot powered by Gemini.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from config import GEMINI_API_KEY, GEMINI_MODEL, TERM_COLUMNS, TERM_MEANINGS
from data_engine import load_scored, verify_against_submission, build_stats_pack, compute_runtime_verification
from tools import (init_tools, get_customer_profile, get_distribution,
                   term_contribution_summary, set_user_corpus_enabled)
from llm import chat_with_gemini, load_context
from rag.ingest import (SUPPORTED_EXTS, build_core, clear_user_corpus,
                        delete_user_file, ingest_user_file, list_user_files,
                        user_store)

# ═══════════════════════════════════════════════════════════════════════════
#  Page config
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ProfitLens — Premier Card Profitability Chatbot",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(88,166,255,0.15);
}
.badge-ok { background: linear-gradient(135deg, #1a472a, #2d6a4f); color: #52c41a;
    padding: 12px 16px; border-radius: 12px; border: 1px solid rgba(82,196,26,0.3);
    font-weight: 600; margin-bottom: 12px; }
.badge-fail { background: linear-gradient(135deg, #4a1a1a, #6a2d2d); color: #ff4d4f;
    padding: 12px 16px; border-radius: 12px; border: 1px solid rgba(255,77,79,0.3);
    font-weight: 600; margin-bottom: 12px; }
.stat-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 16px; margin-bottom: 8px; }
.stat-label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 4px; }
.stat-value { color: #e6edf3; font-size: 1.3rem; font-weight: 700; }
.example-btn { display: block; width: 100%; text-align: left; padding: 10px 14px;
    margin: 4px 0; background: rgba(88,166,255,0.08); border: 1px solid rgba(88,166,255,0.15);
    border-radius: 10px; color: #58a6ff; cursor: pointer; font-size: 0.82rem;
    transition: all 0.2s; }
.example-btn:hover { background: rgba(88,166,255,0.18); border-color: rgba(88,166,255,0.35); }
h1 { background: linear-gradient(135deg, #58a6ff, #bc8cff, #f778ba);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800 !important; }
.stChatMessage { border-radius: 16px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  Data loading (cached)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading & scoring 500K customers…")
def _load_data():
    df = load_scored()
    init_tools(df)
    stats = build_stats_pack(df)
    verification = verify_against_submission(df)
    ctx_verification = compute_runtime_verification(df)
    # Pre-build the runtime block (cached here so it's only computed once)
    runtime_block = ctx_verification["block"]
    # Also warm up the context file cache and print token estimate
    load_context(runtime_block)
    return df, stats, verification, ctx_verification

try:
    df, stats_pack, verification, ctx_verification = _load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Failed to load data: {e}")
    df, stats_pack, verification, ctx_verification = None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════
#  RAG index (core built once per source-file change; user index loaded)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Building knowledge base…")
def _load_rag(_df, _verification):
    status = build_core(_df, _verification)
    n_user = len(user_store())
    print(f"[ProfitLens RAG] User corpus: {n_user:,} chunks "
          f"from {len(list_user_files())} file(s).")
    return status

rag_status, rag_error = None, None
if data_loaded:
    try:
        rag_status = _load_rag(df, verification)
    except Exception as e:
        rag_error = str(e)

# ═══════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💳 ProfitLens")
    st.caption(f"Model: `{GEMINI_MODEL}`")

    if not GEMINI_API_KEY:
        st.error("⚠️ **API key missing!**\n\nAdd `GEMINI_API_KEY=...` or `API_KEY=...` to your `.env` file in the project root.")

    # xlsx integrity badge
    if verification:
        css = "badge-ok" if verification["ok"] else "badge-fail"
        st.markdown(f'<div class="{css}">{verification["message"]}</div>', unsafe_allow_html=True)

    # Context-file missing warning
    from config import CONTEXT_MD_PATH as _ctx_path
    if data_loaded and not _ctx_path.exists():
        st.markdown('<div class="badge-fail">⚠️ Context file missing — answers will be less grounded</div>',
                    unsafe_allow_html=True)

    # Context stats verification badge
    if ctx_verification:
        if ctx_verification["all_match"]:
            st.markdown('<div class="badge-ok">✓ Context stats verified</div>', unsafe_allow_html=True)
        else:
            mlist = ", ".join(ctx_verification["mismatches"])
            st.markdown(f'<div class="badge-fail">⚠️ Context stats stale: {mlist}</div>',
                        unsafe_allow_html=True)

    # Knowledge-base status + user-corpus toggle (visible on both tabs)
    n_user_chunks = len(user_store()) if data_loaded else 0
    if rag_error:
        st.markdown(f'<div class="badge-fail">⚠️ Knowledge base unavailable: {rag_error}</div>',
                    unsafe_allow_html=True)
    elif rag_status:
        st.markdown(f'<div class="badge-ok">📚 Knowledge base: {rag_status["chunks"]} core '
                    f'+ {n_user_chunks} of your chunks</div>', unsafe_allow_html=True)

    include_user = st.toggle(
        "Include my uploaded files in answers",
        value=n_user_chunks > 0,
        disabled=n_user_chunks == 0,
        help="When off, retrieval is restricted to the built-in ProfitLens corpus.",
    )
    set_user_corpus_enabled(include_user)

    free_tier_mode = st.toggle(
        "🆓 Free tier mode",
        value=st.session_state.get("free_tier_mode", True),
        help="On: 1 direct model call per message, no tool-calling loop or knowledge-base "
             "retrieval — stays within Gemini's free-tier daily quota, but answers aren't "
             "grounded in live tool data and charts won't render. Off: full pipeline "
             "(multi-round tool calls + retrieval) for grounded, accurate answers, but "
             "uses more requests per message.",
    )
    st.session_state["free_tier_mode"] = free_tier_mode

    # Dataset stats
    if stats_pack:
        sp = stats_pack
        ps = sp["profit_stats"]
        st.markdown("### 📊 Dataset")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Customers</div>'
                        f'<div class="stat-value">{sp["row_count"]:,}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Top-20% Cutoff</div>'
                        f'<div class="stat-value">${sp["top20_cutoff"]:,.2f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-card"><div class="stat-label">Top-20% Mean Profit</div>'
                    f'<div class="stat-value">${ps["top20"]["mean"]:,.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Try asking")

    EXAMPLES = [
        "Profit breakdown for customer 1",
        "Why is customer 20 in the top 20%?",
        "Compare lounge usage between top 20% and rest",
        "Show the profit distribution",
        "Which cost term hurts profitability the most?",
        "Explain my profitability formula term by term",
    ]
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["pending_example"] = ex

    st.markdown("---")
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["history"] = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
#  Title
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("# ProfitLens")
st.caption("Premier Card Profitability Chatbot — powered by Gemini")

# ═══════════════════════════════════════════════════════════════════════════
#  Session state
# ═══════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []

# ═══════════════════════════════════════════════════════════════════════════
#  Chart rendering helpers
# ═══════════════════════════════════════════════════════════════════════════

def render_waterfall(profile: dict):
    """Render a horizontal waterfall-style bar for a customer's P&L terms."""
    terms = profile.get("terms", {})
    if not terms:
        return
    labels = list(terms.keys())
    values = list(terms.values())
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
    meanings = [TERM_MEANINGS.get(l, l) for l in labels]
    fig = go.Figure(go.Bar(
        y=[TERM_MEANINGS.get(l, l) for l in labels],
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"${v:,.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=f"Customer {profile.get('customer_id', '?')} — P&L Breakdown (Profit: ${profile.get('profit', 0):,.2f})",
        xaxis_title="USD ($)",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=220, r=80),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_histogram(dist_data: dict):
    """Render a plotly histogram from get_distribution output."""
    edges = dist_data.get("bin_edges", [])
    counts = dist_data.get("counts", [])
    if not edges or not counts:
        return
    mids = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
    fig = go.Figure(go.Bar(
        x=mids, y=counts,
        width=[(edges[i+1] - edges[i]) * 0.9 for i in range(len(counts))],
        marker_color="rgba(88,166,255,0.7)",
        marker_line=dict(color="rgba(88,166,255,1)", width=0.5),
    ))
    col = dist_data.get("column", "")
    desc = dist_data.get("description", col)
    fig.update_layout(
        title=f"Distribution of {desc} ({col})",
        xaxis_title=col, yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_term_comparison(tcs: dict):
    """Render grouped bars for term contribution summary."""
    terms_list = tcs.get("terms", [])
    if not terms_list:
        return
    names = [t["meaning"][:30] for t in terms_list]
    top20_vals = [t["top20_mean"] for t in terms_list]
    rest_vals = [t["rest_mean"] for t in terms_list]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Top 20%", y=names, x=top20_vals, orientation="h",
                         marker_color="rgba(46,204,113,0.8)"))
    fig.add_trace(go.Bar(name="Bottom 80%", y=names, x=rest_vals, orientation="h",
                         marker_color="rgba(231,76,60,0.6)"))
    fig.update_layout(
        title="Mean P&L Term: Top 20% vs Rest",
        barmode="group", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=450, margin=dict(l=220),
        yaxis=dict(autorange="reversed"),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════════════════
tab_chat, tab_digest = st.tabs(["💬 Chat", "📂 Digest your own files?"])

with tab_chat:
    # ═══════════════════════════════════════════════════════════════════════════
    #  Render chat history
    # ═══════════════════════════════════════════════════════════════════════════
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-render charts if stored
            if "chart_type" in msg:
                if msg["chart_type"] == "waterfall":
                    render_waterfall(msg["chart_data"])
                elif msg["chart_type"] == "histogram":
                    render_histogram(msg["chart_data"])
                elif msg["chart_type"] == "term_comparison":
                    render_term_comparison(msg["chart_data"])

    # ═══════════════════════════════════════════════════════════════════════════
    #  Chat input + response
    # ═══════════════════════════════════════════════════════════════════════════
    user_input = st.chat_input("Ask about customer profitability…")

    # Handle example button clicks
    if "pending_example" in st.session_state:
        user_input = st.session_state.pop("pending_example")

    if user_input and data_loaded:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    _runtime_block = ctx_verification["block"] if ctx_verification else ""
                    reply, new_history, tools_used = chat_with_gemini(
                        user_input, st.session_state["history"], _runtime_block,
                        free_tier=st.session_state.get("free_tier_mode", True),
                    )
                    st.session_state["history"] = new_history
                    st.markdown(reply)

                    # Determine if we should render charts
                    msg_entry = {"role": "assistant", "content": reply}

                    # Auto-render charts based on tools called
                    if "get_customer_profile" in tools_used:
                        # Try to extract customer id from the user message
                        import re
                        ids = re.findall(r'\b(\d+)\b', user_input)
                        if ids:
                            profile = get_customer_profile(int(ids[0]))
                            if "error" not in profile:
                                render_waterfall(profile)
                                msg_entry["chart_type"] = "waterfall"
                                msg_entry["chart_data"] = profile

                    if "get_distribution" in tools_used:
                        # Try to extract column from user message
                        col = "profit"  # default
                        for c in TERM_COLUMNS + [f"f{i}" for i in range(1, 24)] + ["profit"]:
                            if c in user_input.lower():
                                col = c
                                break
                        dist = get_distribution(col)
                        if "error" not in dist:
                            render_histogram(dist)
                            msg_entry["chart_type"] = "histogram"
                            msg_entry["chart_data"] = dist

                    if "term_contribution_summary" in tools_used:
                        tcs = term_contribution_summary()
                        if "error" not in tcs:
                            render_term_comparison(tcs)
                            msg_entry["chart_type"] = "term_comparison"
                            msg_entry["chart_data"] = tcs

                    st.session_state["messages"].append(msg_entry)

                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error: {e}")

    elif user_input and not data_loaded:
        st.error("Data not loaded. Check that the CSV file exists in the project root.")


# ═══════════════════════════════════════════════════════════════════════════
#  Digest tab — user file ingestion
# ═══════════════════════════════════════════════════════════════════════════
with tab_digest:
    st.markdown("## Do you want to digest your own files?")
    st.caption("Upload documents and ProfitLens will read them and answer "
               "questions about them alongside the built-in analysis.")

    uploads = st.file_uploader(
        "Drop files here",
        type=SUPPORTED_EXTS,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.get('uploader_round', 0)}",
    )

    if uploads:
        for up in uploads:
            with st.spinner(f"Digesting {up.name}…"):
                result = ingest_user_file(up.name, up.getvalue())
            if not result["ok"]:
                st.toast(f"⚠️ {result['message']}", icon="⚠️")
            elif result.get("duplicate"):
                st.toast(f"↩︎ {up.name} already digested", icon="↩️")
            else:
                st.toast(result["message"], icon="✅")
        # Reset the widget so the same files aren't re-processed on rerun.
        st.session_state["uploader_round"] = st.session_state.get("uploader_round", 0) + 1
        st.rerun()

    st.markdown("---")
    st.markdown("### 📑 Your digested files")

    files = list_user_files()
    if not files:
        st.info("No files digested yet. Anything you upload stays local — "
                "only the text chunks are embedded.")
    else:
        header = st.columns([3, 1, 1, 2, 1])
        for col, label in zip(header, ["File", "Type", "Chunks", "Ingested", ""]):
            col.markdown(f"**{label}**")
        for entry in files:
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 2, 1])
            c1.write(entry["filename"])
            c2.write(f".{entry['type']}")
            c3.write(f"{entry['chunks']}")
            c4.write(entry["ingested_at"])
            if c5.button("🗑", key=f"del_{entry['filename']}",
                         help=f"Remove {entry['filename']}"):
                removed = delete_user_file(entry["filename"])
                st.toast(f"Removed {entry['filename']} ({removed} chunks)", icon="🗑️")
                st.rerun()

        st.markdown("")
        if st.session_state.get("confirm_clear"):
            st.warning("Remove every file you've digested? This can't be undone.")
            cc1, cc2 = st.columns(2)
            if cc1.button("Yes, clear all", type="primary", use_container_width=True):
                removed = clear_user_corpus()
                st.session_state["confirm_clear"] = False
                st.toast(f"Cleared {removed} chunks", icon="🧹")
                st.rerun()
            if cc2.button("Cancel", use_container_width=True):
                st.session_state["confirm_clear"] = False
                st.rerun()
        elif st.button("🧹 Clear all my files"):
            st.session_state["confirm_clear"] = True
            st.rerun()

    st.markdown("---")
    st.caption(
        f"Supported: {', '.join('.' + e for e in SUPPORTED_EXTS)} · max 25 MB per file. "
        "Spreadsheets and CSVs are digested as profile documents (schema, per-column "
        "statistics and a short preview) rather than row by row. "
        "Ask about them in the **Chat** tab — answers cite the filename and location."
    )
