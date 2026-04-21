"""
AskSG — Streamlit chat interface.

Run from project root:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing rag.py from the same directory
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from rag import answer, load_retriever

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AskSG",
    page_icon="🇸🇬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Load retriever (cached — runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading document index...")
def get_retriever():
    return load_retriever()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_sources(sources: list[dict]) -> str:
    """Deduplicated source list as markdown."""
    seen: set[tuple] = set()
    lines = []
    for s in sources:
        key = (s["source"], s["document"])
        if key not in seen:
            seen.add(key)
            doc_label = s["document"].replace("_", " ").title()
            lines.append(f"- **{s['source'].upper()}** — {doc_label}")
    return "\n".join(lines)


EXAMPLE_QUESTIONS = [
    "What did Budget 2025 say about support for first-time homebuyers?",
    "Am I eligible to buy an HDB resale flat as a Singapore PR?",
    "What are the CPF contribution rates for employees above 55?",
    "What is MAS's current assessment of Singapore's inflation outlook?",
    "How has Singapore's fiscal stance changed from Budget 2023 to 2025?",
]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("AskSG 🇸🇬")
st.caption(
    "Ask questions about Singapore's Budget, HDB policies, CPF rules, and "
    "monetary policy — answered from official government documents."
)

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show example questions only on a fresh session
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(1)
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                st.markdown(format_sources(msg["sources"]))

# Handle a query — either from example button or chat input
query: str | None = st.session_state.pop("pending_query", None)
if typed := st.chat_input("Ask a question about Singapore policy..."):
    query = typed

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                model, collection = get_retriever()
                result = answer(query, model, collection)

                st.markdown(result["answer"])
                with st.expander("Sources", expanded=False):
                    st.markdown(format_sources(result["sources"]))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

            except FileNotFoundError as e:
                msg = f"**Setup required:** {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            except KeyError:
                msg = "**Missing API key.** Add `GROQ_API_KEY=your_key` to your `.env` file."
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            except Exception as e:
                msg = f"**Error:** {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
