"""
Capstone_streamlit.py — Course Assistant Streamlit UI
Run: streamlit run Capstone_streamlit.py

All expensive resources (LLM, ChromaDB, compiled graph) are initialised
ONCE inside @st.cache_resource and reused on every Streamlit rerun.
This is the fix for the blank-page / reloading bug.
"""

import uuid
import streamlit as st

st.set_page_config(
    page_title="Course Assistant — Agentic AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── All heavy init goes here — runs ONCE per server process ─────────────
@st.cache_resource
def load_resources():
    """
    Build the knowledge base and compile the LangGraph agent.
    @st.cache_resource guarantees this runs only once, no matter how many
    times the user sends a message or the page reruns.
    """
    from Agent import build_knowledge_base, build_graph, DOCUMENTS
    collection   = build_knowledge_base()
    compiled_app = build_graph(collection)
    return compiled_app, DOCUMENTS


# Attempt to load; show a friendly error if GROQ_API_KEY is missing
load_error = None
compiled_app = None
DOCUMENTS = []

try:
    compiled_app, DOCUMENTS = load_resources()
except Exception as exc:
    load_error = str(exc)


# ── Session state — persists across reruns ───────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Agent runner ─────────────────────────────────────────────────────────
def run_agent(question: str) -> dict:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {
        "question":     question,
        "messages":     [],
        "route":        "",
        "retrieved":    "",
        "sources":      [],
        "tool_result":  "",
        "answer":       "",
        "faithfulness": 0.0,
        "eval_retries": 0,
        "user_name":    "",
    }
    return compiled_app.invoke(initial_state, config=config)


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Course Assistant")
    st.markdown("**Agentic AI — 13-Day Course**")
    st.markdown("_Dr. Kanthi Kiran Sirra_")
    st.divider()

    st.markdown("### 📚 Topics Covered")
    for d in DOCUMENTS:
        st.markdown(f"- {d['topic']}")

    st.divider()
    st.markdown("### 💡 Try asking")
    for sample in [
        "What is LangGraph?",
        "How does MemorySaver work?",
        "Explain the RAG pipeline",
        "What was covered on Day 5?",
        "What is the eval node?",
        "What is today's date?",
    ]:
        st.markdown(f"• _{sample}_")

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages  = []
        st.rerun()

    st.caption(f"Session: `{st.session_state.thread_id[:8]}…`")

    with st.expander("🔧 Architecture"):
        st.markdown("""
**8 Nodes:**
1. `memory_node`
2. `router_node`
3. `retrieval_node`
4. `skip_retrieval_node`
5. `tool_node`
6. `answer_node`
7. `eval_node`
8. `save_node`

**Routes:** retrieve · memory_only · tool

**Stack:** LangGraph · ChromaDB · Groq (llama-3.3-70b) · Streamlit
        """)


# ── Main area ────────────────────────────────────────────────────────────
st.title("🤖 Agentic AI Course Assistant")
st.markdown(
    "Ask me anything about the **13-day Agentic AI course**. "
    "Answers are strictly grounded in the course knowledge base."
)

# Show error banner if load failed
if load_error:
    st.error(f"❌ Failed to load agent: {load_error}")
    st.info(
        "Make sure your `.env` file contains `GROQ_API_KEY=<your-key>` "
        "then restart with `streamlit run Capstone_streamlit.py`."
    )
    st.stop()

# Re-render chat history on every rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")
            if msg.get("faithfulness") is not None:
                f     = msg["faithfulness"]
                color = "green" if f >= 0.7 else "orange"
                st.markdown(
                    f"<small style='color:{color}'>Faithfulness: {f:.2f}</small>",
                    unsafe_allow_html=True,
                )

# Welcome message when no conversation yet
if not st.session_state.messages:
    st.info(
        "👋 Welcome! Try asking: _What is LangGraph?_ "
        "or _How does the eval node work?_"
    )

# Chat input — this is the ONLY place that triggers a new agent call
if prompt := st.chat_input("Ask about the Agentic AI course…"):

    # Append and show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent and show the response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result      = run_agent(prompt)
                answer      = result.get("answer", "Sorry, I could not generate a response.")
                sources     = result.get("sources", [])
                faithfulness = result.get("faithfulness")
                route       = result.get("route", "")
            except Exception as exc:
                answer      = f"⚠️ Error: {exc}"
                sources     = []
                faithfulness = None
                route       = ""

        st.markdown(answer)

        if sources:
            with st.expander("📎 Sources"):
                for src in sources:
                    st.markdown(f"- {src}")

        if faithfulness is not None and route == "retrieve":
            color = "green" if faithfulness >= 0.7 else "orange"
            st.markdown(
                f"<small style='color:{color}'>Faithfulness: {faithfulness:.2f}</small>",
                unsafe_allow_html=True,
            )

    # Persist assistant message in session state
    st.session_state.messages.append({
        "role":        "assistant",
        "content":     answer,
        "sources":     sources,
        "faithfulness": faithfulness if route == "retrieve" else None,
    })
