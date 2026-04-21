"""
Agent.py — Course Assistant: Agentic AI (13-Day Course)
Pure Python embeddings — works on Python 3.13, no torch, no onnx, no DLL needed.

HOW TO USE:
  from Agent import build_graph, build_knowledge_base, ask, DOCUMENTS
  collection = build_knowledge_base()
  app = build_graph()
"""

import os
import re
import math
import hashlib
import datetime
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb

load_dotenv()

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2

# ─────────────────────────────────────────────
#  LLM — lazy init so import doesn't fail if key missing
# ─────────────────────────────────────────────
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ─────────────────────────────────────────────
#  PURE PYTHON EMBEDDER — no torch, no onnx
#  Works on Python 3.13 with zero DLL issues
# ─────────────────────────────────────────────
def _embed_text(text: str) -> list:
    """256-dim embedding via character n-gram hashing. Pure Python."""
    vec = [0.0] * 256
    text = text.lower()
    for n in (2, 3, 4):
        for i in range(len(text) - n + 1):
            gram = text[i : i + n]
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            vec[h % 256] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed(texts: list) -> list:
    return [_embed_text(t) for t in texts]


# ─────────────────────────────────────────────
#  KNOWLEDGE BASE — 13 documents, one topic each
# ─────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "LangGraph Overview",
        "text": """LangGraph is a library built on top of LangChain that enables developers to
build stateful multi-actor applications with Large Language Models. Unlike simple chain-based
pipelines, LangGraph models agent behaviour as a directed graph where each node represents a
step of computation and edges represent the flow of control between steps.

The central design principle of LangGraph is that state is explicit. Every piece of information
that flows through the graph lives in a TypedDict called State. Nodes read from State and write
updates back. The graph runtime merges those updates automatically.

LangGraph supports conditional edges which allow the graph to branch based on the current State.
For example after a router node decides whether to retrieve documents or call a tool a conditional
edge reads state.route and dispatches to the correct next node.

LangGraph integrates with MemorySaver a built-in checkpointing mechanism that persists the full
graph state between invocations using a thread_id. This enables true multi-turn conversations.

Key use cases include RAG agents tool-using agents multi-agent orchestration self-correcting
pipelines and stateful chatbots.""",
    },
    {
        "id": "doc_002",
        "topic": "StateGraph and Node Design",
        "text": """A StateGraph is the core abstraction in LangGraph. It is created by instantiating
StateGraph with your State class. The State class must be a TypedDict and it defines every field
that nodes can read or write.

Nodes are plain Python functions with the signature: def my_node(state) returning a dict. They
receive the current State as input and return a dictionary of only the fields they want to update.

Edges connect nodes. Fixed edges always route from one node to another. Conditional edges call a
Python function that inspects the State and returns the name of the next node as a string.

Every graph needs an entry point set with set_entry_point and every terminal node must connect to
END. The most common compile error is forgetting the final save to END edge.

Best practice: design your State TypedDict completely BEFORE writing any node function. Adding
fields after nodes are written forces updates to every affected node. Think of State as a database
schema. Changing it is expensive.""",
    },
    {
        "id": "doc_003",
        "topic": "MemorySaver and Conversation Memory",
        "text": """MemorySaver is LangGraph's built-in checkpointer that enables persistent multi-turn
conversation. LLMs themselves are stateless and have zero memory of previous API calls. MemorySaver
solves this by serialising and storing the complete graph State after every invoke call keyed by a
thread_id.

Usage: compile the graph with checkpointer=MemorySaver(). Then pass a config dictionary on every
invoke call: configurable with thread_id set to some string value. LangGraph automatically restores
the State associated with that thread_id before executing the graph and saves the updated State
afterwards.

thread_id can be any string. Common choices are a UUID a username or a session ID. Using the same
thread_id across multiple invoke calls gives the agent full context of the previous conversation.
Using a new thread_id starts a fresh conversation.

Sliding window: on long conversations the full message history can exceed the LLM context window.
A common mitigation is to keep only the last 6 messages in the messages list inside the State.
The history is trimmed in the memory_node before being passed to the answer_node.

In Streamlit st.session_state stores the thread_id across reruns. A New Conversation button
resets thread_id to a new UUID and clears the displayed chat history.""",
    },
    {
        "id": "doc_004",
        "topic": "ChromaDB and Vector Retrieval",
        "text": """ChromaDB is an open-source embedding database used for storing and querying document
embeddings. In the Agentic AI course ChromaDB is used as the vector store backing the RAG component.

Setup: create an in-memory client with chromadb.Client(), create a collection with
client.create_collection, and add documents with collection.add(). The add call takes documents
embeddings ids and metadatas as arguments.

Retrieval: embed the query using the same embedding function used during ingestion then call
collection.query with query_embeddings and n_results set to 3. The result contains documents and
metadatas lists at index zero.

Key rule: ALWAYS test retrieval before building the graph. A broken knowledge base cannot be fixed
by improving the LLM prompt. If retrieval returns irrelevant documents rewrite the documents.

The all-MiniLM-L6-v2 model produces 384-dimensional vectors. It runs locally on CPU with no API
key required making it suitable for offline development. In this project we use a pure Python
embedder that produces 256-dimensional vectors with no external dependencies.""",
    },
    {
        "id": "doc_005",
        "topic": "RAG Retrieval-Augmented Generation",
        "text": """RAG stands for Retrieval-Augmented Generation. It is the technique of combining a
retrieval step with a generation step. The retrieval step finds relevant documents from a knowledge
base. The generation step uses an LLM to synthesise an answer from those documents.

Without RAG LLMs answer questions from their training data which may be outdated or prone to
hallucination. With RAG the LLM is given only the retrieved context as grounding material and
instructed to answer only from that context. This dramatically reduces hallucination.

The pipeline in the course: user question then embed question then query ChromaDB then top-k chunks
returned then chunks formatted into context string then passed to LLM in system prompt.

The system prompt must include an explicit grounding rule such as: Answer ONLY from the provided
context. If the answer is not in the context say I don't have that information in my knowledge base.
This rule is non-negotiable for faithful answers.

Sources: after retrieval the topic names of the retrieved chunks are stored in state.sources. These
are displayed in the Streamlit UI as citation references helping users verify the origin of answers.

RAG limitation: if the question is not covered by any document the agent must admit it does not
know rather than fabricate an answer.""",
    },
    {
        "id": "doc_006",
        "topic": "Router Node and Routing Logic",
        "text": """The router_node decides which path the graph should take after receiving a user
question. It uses an LLM prompt to classify the question into one of three routes: retrieve,
memory_only, or tool.

retrieve: the question requires information from the knowledge base. Use this when the user asks
about course topics concepts tools or specific session content.

memory_only: the question can be answered from conversation history without any retrieval or tool
call. Use this for meta-questions like what did you just say or can you repeat that.

tool: the question requires live computation outside the knowledge base. Use this when the user asks
about the current date time or day of the week.

The router prompt must describe each route explicitly with examples. Vague routing prompts cause
incorrect routing decisions. The router should reply with ONLY one word.

The route_decision function reads state.route and returns the name of the next node as a string
which LangGraph uses for the conditional edge dispatch after router_node.""",
    },
    {
        "id": "doc_007",
        "topic": "Tool Node and Datetime Tool",
        "text": """Tools in LangGraph agents handle tasks that the knowledge base cannot: current date
and time arithmetic live web data and API calls. The tool_node implements tool execution inside the
graph.

Golden rule: tools must NEVER raise exceptions. A crashing tool crashes the entire graph run and
produces an unhandled error in the UI. Always wrap tool logic in a try except block and return an
error string if the tool fails.

The datetime tool implemented in this project:
Detects what the user is asking for: date time day of the week or all three.
Uses Python datetime.datetime.now() to get the current timestamp.
Returns a human-readable string such as Today is Monday 21 April 2026 and the current time is
14:35:22.
Never raises an exception and any failure returns a graceful error message.

The router_node must be prompted to route datetime and time-related questions to the tool route.
Without this such questions would incorrectly go to the retrieve route.""",
    },
    {
        "id": "doc_008",
        "topic": "Eval Node and Self-Reflection",
        "text": """The eval_node implements self-reflection: the ability of the agent to score the
quality of its own answer and retry if the quality is insufficient.

Faithfulness scoring: after answer_node produces a response eval_node sends the answer and the
retrieved context to the LLM with a prompt asking it to rate whether the answer uses ONLY
information from the context. The score is a float from 0.0 to 1.0. A score of 1.0 means fully
faithful with no hallucination. A score of 0.0 means the answer ignores the context completely.

Retry logic: if the faithfulness score is below FAITHFULNESS_THRESHOLD which defaults to 0.7 then
eval_decision returns answer to trigger a retry. The graph loops back to answer_node which
re-generates the answer with an additional instruction saying the previous answer did not meet
quality standards.

Safety valve: MAX_EVAL_RETRIES defaults to 2 and prevents infinite loops. Once eval_retries reaches
this limit eval_decision returns save regardless of the faithfulness score.

Skip condition: if state.retrieved is empty for memory_only or tool routes eval_node skips the
faithfulness check and returns a score of 1.0 because there is no context to be faithful to.""",
    },
    {
        "id": "doc_009",
        "topic": "Streamlit Deployment",
        "text": """Streamlit is a Python library for building web UIs with minimal code. In the Agentic
AI course it is used to deploy the capstone agent as a chat interface.

The st.cache_resource decorator wraps all expensive initialisations including the LLM the embedder
the ChromaDB collection and the compiled LangGraph app. This ensures they are created only once per
Streamlit server process and reused on every rerun. Without caching the app reloads the model on
every message making it unusably slow.

st.session_state persists variables across reruns. The chat messages list and thread_id must both
be stored in st.session_state.

New Conversation button: clicking this button generates a new thread_id using uuid.uuid4() and
clears st.session_state.messages. This resets the LangGraph MemorySaver context starting a fresh
conversation.

Chat UI: use st.chat_message blocks to display messages. Use st.chat_input to capture user input.
Loop over st.session_state.messages at the top of the script to re-render the full chat history.

Sidebar: add a sidebar with domain description topics covered example questions and the New
Conversation button using st.sidebar.""",
    },
    {
        "id": "doc_010",
        "topic": "Session Topics Days 1 to 5",
        "text": """Day 1 covered the foundations of Agentic AI: what agents are how they differ from
simple LLM calls the concept of a reasoning loop perceive think act and an introduction to the
LangChain ecosystem. Students set up their environment installed LangChain LangGraph and Groq and
ran their first LLM call.

Day 2 introduced tool use. Students built agents that call external tools including web search using
DDGS a calculator using Python eval and a simple API call. The key lesson was that tools must never
raise exceptions and must always return strings.

Day 3 covered prompt engineering for agents: system prompts grounding rules to prevent hallucination
few-shot examples chain-of-thought prompting and how to instruct the LLM to reply in structured
formats like JSON or a single word.

Day 4 introduced LangGraph StateGraph. Students defined their first TypedDict State wrote node
functions connected them with edges and compiled the graph. The mandatory sequence State first then
nodes then edges was emphasised strongly.

Day 5 covered RAG fundamentals: document chunking embedding with SentenceTransformer building a
ChromaDB collection querying by embedding similarity and integrating retrieval into a LangGraph
node. Students built and tested their first RAG-enabled agent.""",
    },
    {
        "id": "doc_011",
        "topic": "Session Topics Days 6 to 10",
        "text": """Day 6 deepened RAG knowledge: metadata filtering in ChromaDB multi-query retrieval
re-ranking results and handling the case where retrieval returns irrelevant documents. Students
learned that documents must be specific and that vague documents produce vague answers.

Day 7 covered conversation memory. Students implemented MemorySaver multi-turn conversation with
thread_id sliding window truncation and tested that a follow-up question requiring context from
Turn 1 is answered correctly without the context being re-stated.

Day 8 introduced self-reflection: the eval node pattern where the LLM scores its own answer for
faithfulness against the retrieved context. Students implemented retry logic with MAX_EVAL_RETRIES
to prevent infinite loops.

Day 9 covered advanced tool use: datetime queries web search integration with DDGS a calculator
tool and a weather API call. The router node was extended to route tool questions correctly.

Day 10 was a full integration day. Students assembled all components including RAG memory tools and
eval into a single compiled LangGraph graph and ran end-to-end tests with ten diverse questions
covering different routes and edge cases.""",
    },
    {
        "id": "doc_012",
        "topic": "Session Topics Days 11 to 13 and RAGAS Evaluation",
        "text": """Day 11 introduced RAGAS an evaluation framework for RAG systems. RAGAS computes three
metrics: faithfulness which measures whether the answer uses only the context, answer_relevancy
which measures whether the answer is relevant to the question, and context_precision which measures
whether the retrieved chunks are relevant. Students ran a baseline evaluation with five QA pairs.

Day 12 covered Streamlit deployment in depth: st.cache_resource st.session_state the New
Conversation button with UUID reset sidebar design and how to display sources alongside the answer.
Students also learned FastAPI as an alternative deployment target.

Day 13 was the Capstone day. Students chose a domain built a complete production agent from scratch
following the 8-part process: knowledge base then State design then node functions then graph
assembly then testing then RAGAS evaluation then Streamlit deployment then written summary.

RAGAS metrics reference: faithfulness above 0.8 is excellent. Between 0.7 and 0.8 is good. Below
0.7 requires knowledge base improvement or prompt refinement. Answer relevancy above 0.75 is the
target. The capstone submission requires three files: the completed notebook capstone_streamlit.py
and Agent.py.""",
    },
    {
        "id": "doc_013",
        "topic": "Groq API and LLM Configuration",
        "text": """Groq is a cloud inference platform offering very fast LLM inference on custom LPU
hardware. In the Agentic AI course Groq is used as the LLM backend because it is free for students
and extremely fast typically producing 500 or more tokens per second on llama-3.3-70b-versatile.

Setup: create a free account at console.groq.com generate an API key and add it to a .env file as
GROQ_API_KEY equals your key. Load it with python-dotenv using load_dotenv() followed by
os.getenv(\"GROQ_API_KEY\").

Model: llama-3.3-70b-versatile is the recommended model for this course. It is a 70 billion
parameter instruction-tuned Llama model. Use temperature=0 for all agent tasks requiring consistent
deterministic outputs such as routing eval scoring and grounded answering.

ChatGroq usage: from langchain_groq import ChatGroq. Instantiate with ChatGroq using the model name
and temperature=0. Invoke with llm.invoke(messages) where messages is a list of LangChain message
objects. The response is an AIMessage and access its text with response.content.

Rate limits on the free Groq tier: the free tier supports 30 requests per minute and 6000 tokens
per minute. Implement a sliding window in memory_node to limit token usage. If you hit rate limits
add time.sleep(2) between agent calls in batch tests.""",
    },
]


# ─────────────────────────────────────────────
#  BUILD CHROMADB COLLECTION
# ─────────────────────────────────────────────
def build_knowledge_base():
    """Build and return a ChromaDB collection loaded with all course documents."""
    client = chromadb.Client()
    try:
        client.delete_collection("course_assistant_kb")
    except Exception:
        pass

    collection = client.create_collection("course_assistant_kb")

    texts = [d["text"] for d in DOCUMENTS]
    ids = [d["id"] for d in DOCUMENTS]
    embeddings = embed(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )
    print(f"✅ Knowledge base ready: {collection.count()} documents")
    return collection


# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:     str
    messages:     List[dict]
    route:        str
    retrieved:    str
    sources:      List[str]
    tool_result:  str
    answer:       str
    faithfulness: float
    eval_retries: int
    user_name:    str


# ─────────────────────────────────────────────
#  NODE FUNCTIONS
# ─────────────────────────────────────────────
def memory_node(state: CapstoneState) -> dict:
    msgs = list(state.get("messages", []))
    msgs.append({"role": "user", "content": state["question"]})
    if len(msgs) > 6:
        msgs = msgs[-6:]
    user_name = state.get("user_name", "")
    match = re.search(r"my name is ([A-Za-z]+)", state["question"], re.IGNORECASE)
    if match:
        user_name = match.group(1).strip().title()
    return {"messages": msgs, "user_name": user_name}


def router_node(state: CapstoneState) -> dict:
    llm = get_llm()
    question = state["question"]
    messages = state.get("messages", [])
    recent = "; ".join(
        f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]
    ) or "none"

    prompt = f"""You are a router for a Course Assistant chatbot about a 13-day Agentic AI course.

Routes:
- retrieve: user asks about course content, LangGraph, ChromaDB, RAG, MemorySaver, Streamlit, Groq, sessions, nodes, tools, eval, RAGAS, or any course topic.
- memory_only: user refers to something already said, like "what did you just say?" or "repeat that".
- tool: user asks about current date, time, or day of the week.

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    if "memory" in decision:
        decision = "memory_only"
    elif "tool" in decision:
        decision = "tool"
    else:
        decision = "retrieve"
    return {"route": decision}


def retrieval_node(state: CapstoneState) -> dict:
    # collection is passed via closure from build_graph()
    raise RuntimeError("retrieval_node must be created via make_retrieval_node(collection)")


def make_retrieval_node(collection):
    """Factory: returns a retrieval_node closure bound to the given collection."""
    def _retrieval_node(state: CapstoneState) -> dict:
        q_emb = embed([state["question"]])
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": topics}
    return _retrieval_node


def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}


def tool_node(state: CapstoneState) -> dict:
    try:
        now = datetime.datetime.now()
        day_name = now.strftime("%A")
        date_str = now.strftime("%d %B %Y")
        time_str = now.strftime("%H:%M:%S")
        q = state["question"].lower()
        if any(w in q for w in ["time", "clock", "hour", "minute"]):
            result = f"The current time is {time_str}."
        elif any(w in q for w in ["day", "weekday"]):
            result = f"Today is {day_name}, {date_str}."
        else:
            result = f"Today is {day_name}, {date_str}. The current time is {time_str}."
    except Exception as e:
        result = f"Datetime tool error: {e}"
    return {"tool_result": result}


def answer_node(state: CapstoneState) -> dict:
    llm = get_llm()
    question     = state["question"]
    retrieved    = state.get("retrieved", "")
    tool_result  = state.get("tool_result", "")
    messages     = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)
    user_name    = state.get("user_name", "")

    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"TOOL RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    name_note = f" The user's name is {user_name}." if user_name else ""

    if context:
        system_content = (
            f"You are a helpful Course Assistant for a 13-day Agentic AI course "
            f"taken by B.Tech 4th year students.{name_note}\n\n"
            "STRICT RULE: Answer ONLY from the provided context below.\n"
            "If the answer is not found in the context, say exactly: "
            "I don't have that information in my knowledge base.\n"
            "Do NOT add any information from your training data.\n"
            "Do NOT guess or speculate.\n\n"
            f"{context}"
        )
    else:
        system_content = (
            f"You are a helpful Course Assistant for a 13-day Agentic AI course.{name_note}\n"
            "Answer based on the conversation history only."
        )

    if eval_retries > 0:
        system_content += (
            "\n\nIMPORTANT: Your previous answer did not pass the faithfulness check. "
            "Answer using ONLY information explicitly stated in the context above."
        )

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        if msg["role"] == "user":
            lc_msgs.append(HumanMessage(content=msg["content"]))
        else:
            lc_msgs.append(AIMessage(content=msg["content"]))
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}


def eval_node(state: CapstoneState) -> dict:
    llm = get_llm()
    answer  = state.get("answer", "")
    context = state.get("retrieved", "")[:500]
    retries = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = (
        "Rate faithfulness: does this answer use ONLY information from the context?\n"
        "Reply with ONLY a decimal number between 0.0 and 1.0.\n"
        "1.0 = fully faithful. 0.0 = mostly hallucinated.\n\n"
        f"Context: {context}\nAnswer: {answer[:300]}"
    )
    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.5

    gate = "PASS" if score >= FAITHFULNESS_THRESHOLD else "RETRY"
    print(f"  [eval] Faithfulness: {score:.2f} {gate}")
    return {"faithfulness": score, "eval_retries": retries + 1}


def save_node(state: CapstoneState) -> dict:
    msgs = list(state.get("messages", []))
    msgs.append({"role": "assistant", "content": state.get("answer", "")})
    return {"messages": msgs}


# ─────────────────────────────────────────────
#  EDGE FUNCTIONS
# ─────────────────────────────────────────────
def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "memory_only":
        return "skip"
    elif route == "tool":
        return "tool"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score < FAITHFULNESS_THRESHOLD and retries < MAX_EVAL_RETRIES:
        return "answer"
    return "save"


# ─────────────────────────────────────────────
#  BUILD GRAPH
# ─────────────────────────────────────────────
def build_graph(collection=None):
    """
    Build and compile the LangGraph agent.
    Pass a collection from build_knowledge_base(), or it will build one internally.
    """
    if collection is None:
        collection = build_knowledge_base()

    _retrieval_node = make_retrieval_node(collection)

    graph = StateGraph(CapstoneState)

    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", _retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_edge("save",     END)

    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )
    graph.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"},
    )

    compiled = graph.compile(checkpointer=MemorySaver())
    print("✅ Graph compiled successfully")
    return compiled


# ─────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────
def ask(app, question: str, thread_id: str = "default") -> dict:
    """Run one question through the compiled agent app."""
    config = {"configurable": {"thread_id": thread_id}}
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
    return app.invoke(initial_state, config=config)


# ─────────────────────────────────────────────
#  SMOKE TEST  (python Agent.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("COURSE ASSISTANT — SMOKE TEST")
    print("=" * 55)
    kb = build_knowledge_base()
    compiled_app = build_graph(kb)
    for q in ["What is LangGraph?", "What is today's date?"]:
        print(f"\n❓ {q}")
        res = ask(compiled_app, q, "smoke")
        print(f"   Route  : {res.get('route')}")
        print(f"   Answer : {res.get('answer', '')[:150]}")
