# 🤖 Course Assistant — Agentic AI (13-Day Course)

**Capstone Project | Dr. Kanthi Kiran Sirra | Agentic AI Course 2026**

---

## 📁 File Structure

```
course_assistant/
├── Agent.py                  ← Core agent: KB, nodes, graph (import-safe, no side effects)
├── Capstone_streamlit.py     ← Streamlit UI (run this to launch the app)
├── Day13_capstone.ipynb      ← Capstone notebook: all 8 parts with tests
├── requirements.txt          ← Python dependencies
├── .env.example              ← Copy to .env and add your Groq API key
└── README.md                 ← This file
```

---

## ⚡ Quick Start

### 1. Clone / unzip the project
```bash
cd course_assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
```bash
cp .env.example .env
# Edit .env and replace the placeholder with your real key
# Get a free key at: https://console.groq.com
```

### 4. Launch the Streamlit app
```bash
streamlit run Capstone_streamlit.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 🧠 Architecture

```
User question
     ↓
[memory_node]    → append to history, sliding window (last 6), extract name
     ↓
[router_node]    → LLM classifies: retrieve / memory_only / tool
     ↓
[retrieval_node / skip_retrieval_node / tool_node]
     ↓
[answer_node]    → grounded answer from context + history
     ↓
[eval_node]      → faithfulness score (0.0–1.0), retry if < 0.7
     ↓
[save_node]      → append answer to messages → END
```

**Stack:** LangGraph · ChromaDB · Groq (llama-3.3-70b-versatile) · Streamlit

---

## 🔑 Key Design Decisions

| Decision | Reason |
|---|---|
| Pure-Python embedder (hash n-gram) | Zero dependencies — works on Python 3.13, no torch, no onnx, no DLL issues |
| `@st.cache_resource` in Streamlit | Prevents KB and graph from reloading on every message rerun |
| `build_graph(collection)` factory | Agent.py has no module-level side effects — safe to import anywhere |
| Sliding window (last 6 msgs) | Prevents token overflow on Groq free tier |
| `MAX_EVAL_RETRIES = 2` | Safety valve prevents infinite faithfulness retry loops |

---

## 📋 Submission Checklist

- [x] `Day13_capstone.ipynb` — Kernel > Restart & Run All passes with no errors
- [x] `Capstone_streamlit.py` — launches with `streamlit run Capstone_streamlit.py`
- [x] `Agent.py` — importable module, no top-level side effects

---

## 💡 Example Questions

- *What is LangGraph?*
- *How does MemorySaver work?*
- *Explain the RAG pipeline*
- *What was covered on Day 5?*
- *What is today's date?*
- *What is RAGAS?*
