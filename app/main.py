"""
AskSG — Streamlit chat interface.

Run from project root:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from rag import load_retriever, stream_answer

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
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

/* Submit button next to the input */
div[data-testid="stFormSubmitButton"] > button {
    background-color: #C8102E;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    transition: background 0.15s;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #a00d24;
    color: white;
    border: none;
}

/* Example question buttons */
div[data-testid="stButton"] > button {
    text-align: left !important;
    justify-content: flex-start;
    background: #fafafa;
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    color: #444;
    padding: 0.55rem 1rem;
    font-size: 0.9rem;
    width: 100%;
    transition: border-color 0.15s, color 0.15s, background 0.15s;
}
div[data-testid="stButton"] > button:hover {
    border-color: #C8102E;
    color: #C8102E;
    background: #fff8f8;
}

div[data-testid="stExpander"] {
    border: 1px solid #f0f0f0 !important;
    border-radius: 8px !important;
}

div[data-testid="stChatMessage"] {
    padding: 0.25rem 0;
}

/* Tighten gap between example question buttons */
div[data-testid="stButton"] {
    margin-bottom: -0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load retriever (cached — runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading document index...")
def get_retriever():
    return load_retriever()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """Escape dollar signs so Streamlit doesn't treat them as LaTeX."""
    return text.replace("$", r"\$")


def format_sources(sources: list[dict]) -> str:
    """Deduplicated source list with a short text snippet per source."""
    seen: set[tuple] = set()
    lines = []
    for s in sources:
        key = (s["source"], s["document"])
        if key not in seen:
            seen.add(key)
            doc_label = s["document"].replace("_", " ").title()
            snippet = s["text"][:200].replace("\n", " ").strip()
            if len(s["text"]) > 200:
                snippet += "..."
            lines.append(
                f"**{s['source'].upper()}** — {doc_label}\n"
                f"> {_escape(snippet)}"
            )
    return "\n\n".join(lines)


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

if "messages" not in st.session_state:
    st.session_state.messages = []

# Input bar — always at the top, above examples
with st.form("query_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        typed = st.text_input(
            label="query",
            placeholder="Ask a question about Singapore policy...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Ask", use_container_width=True)

query: str | None = st.session_state.pop("pending_query", None)
if submitted and typed.strip():
    query = typed.strip()

# Example questions — only shown on fresh session
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

# Conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(_escape(msg["content"]))
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                st.markdown(format_sources(msg["sources"]))

# Handle query
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            model, collection = get_retriever()
            groq_stream, chunks = stream_answer(query, model, collection)

            def _token_gen():
                for event in groq_stream:
                    yield _escape(event.choices[0].delta.content or "")

            answer_text = st.write_stream(_token_gen())

            with st.expander("Sources", expanded=False):
                st.markdown(format_sources(chunks))

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": chunks,
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
