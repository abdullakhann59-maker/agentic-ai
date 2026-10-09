"""Streamlit front end for the Nexus agent (talks to the FastAPI backend).

Run:  streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import get_settings  # noqa: E402

SETTINGS = get_settings()
WORKSPACE = Path(SETTINGS.workspace_dir)
TOOL_NAMES = [
    "web_search", "web_read", "browser_open", "browser_find", "browser_click", "browser_type",
    "browser_screenshot", "scrape_analyze", "scrape_api", "scrape_embedded", "scrape_iframes", "scrape_page",
    "scrape_structure", "scrape_save", "doc_read", "ocr_image", "invoice_extract", "data_describe",
    "data_analyze", "chart_make", "email_draft", "email_send", "memory_save", "memory_search", "calculator",
]
EXAMPLES = [
    "What can you do?",
    "Scrape product names and prices from http://localhost:8001/dynamic/",
    "Extract the data from the iframes on http://localhost:8001/iframe/",
    "What is the total revenue by region in sales_data.csv? Make a bar chart too.",
    "Extract the invoice details from sample_invoice.pdf",
    "How many days of paid leave do employees get?",
    "Summarise http://localhost:8001/injection/",
    "Draft an email to priya@example.com with a summary of Q3 results and send it",
]
PATH_LABELS = {
    "quick": ("Quick answer (chat knowledge)", "#16a34a"),
    "playbook": ("Task playbook from vector DB", "#2563eb"),
    "direct": ("Direct LLM (no matching knowledge)", "#9333ea"),
    "confirmation": ("Confirmed action", "#ea580c"),
    "error": ("Error", "#dc2626"),
}

st.set_page_config(page_title="Nexus Agent", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.badge{display:inline-block;padding:2px 10px;border-radius:999px;color:#fff;font-size:0.78rem;margin-right:6px}
.meta{color:#6b7280;font-size:0.8rem}
.block-container{padding-top:3.2rem}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- state
if "session_id" not in st.session_state:
    st.session_state.session_id = "web-" + uuid.uuid4().hex[:6]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "queued" not in st.session_state:
    st.session_state.queued = None


def api() -> str:
    return st.session_state.get("api_url", SETTINGS.api_url).rstrip("/")


@st.cache_data(ttl=15, show_spinner=False)
def get_health(url: str) -> dict:
    try:
        return requests.get(f"{url}/health", timeout=5).json()
    except requests.RequestException:
        return {"status": "api_down"}


def send(message: str, upload=None) -> None:
    st.session_state.messages.append({"role": "user", "content": message,
                                      "file": upload.name if upload else None})
    files = {"file": (upload.name, upload.getvalue())} if upload else None
    with st.spinner("Agent is working... (retrieving knowledge, planning, using tools)"):
        try:
            r = requests.post(f"{api()}/chat", data={"message": message, "session_id": st.session_state.session_id},
                              files=files, timeout=900)
            r.raise_for_status()
            res = r.json()
        except requests.RequestException as e:
            res = {"answer": f"Could not reach the API: {e}", "status": "failed", "path": "error", "steps": [],
                   "tools_used": [], "files": [], "tokens": {"total": 0}, "time_ms": 0, "knowledge_used": []}
    st.session_state.messages.append({"role": "assistant", "content": res["answer"], "res": res})


# -------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🤖 Nexus Agent")
    st.caption("LangGraph agent · RAG with ChromaDB · Ollama · Selenium")
    st.text_input("API URL", value=SETTINGS.api_url, key="api_url")
    h = get_health(api())
    st.markdown("#### System health")
    if h.get("status") == "api_down":
        st.error("API is not running. Start it with: uvicorn app.api:app")
    else:
        def dot(ok: bool) -> str:
            return "🟢" if ok else "🔴"
        st.write(f"{dot(h.get('ollama') == 'ok')} Ollama")
        if h.get("ollama") == "ok":
            st.write(f"{dot(h['llm_model']['available'])} LLM: `{h['llm_model']['name']}`")
            st.write(f"{dot(h['embedding_model']['available'])} Embeddings: `{h['embedding_model']['name']}`")
        vdb = h.get("vector_db", {})
        st.write(f"{dot(vdb.get('status') == 'ok')} Vector DB")
        if vdb.get("items"):
            i = vdb["items"]
            st.caption(f"{i['task']} task playbooks · {i['chat']} chat examples · {i['document']} doc chunks · "
                       f"{i['memory']} memories")
        st.write(f"{dot(h.get('selenium') == 'ok')} Selenium")
    if st.button("🔄 Refresh health"):
        get_health.clear()
        st.rerun()
    st.divider()
    st.markdown("#### Session")
    st.code(st.session_state.session_id, language=None)
    if st.button("➕ New chat"):
        st.session_state.session_id = "web-" + uuid.uuid4().hex[:6]
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("#### Try an example")
    for ex in EXAMPLES:
        if st.button(ex, key="ex_" + ex, use_container_width=True):
            st.session_state.queued = ex

tab_chat, tab_kb, tab_dash = st.tabs(["💬 Agent Chat", "📚 Knowledge Base", "📊 Dashboard"])


# ------------------------------------------------------------- chat tab
def render_files(files: list[str], key: str) -> None:
    for i, f in enumerate(files):
        p = WORKSPACE / f
        if not p.exists():
            st.caption(f"📄 {f}")
            continue
        if p.suffix.lower() == ".png":
            st.image(str(p), caption=f, width=680)
        elif p.suffix.lower() in (".csv", ".xlsx"):
            df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)
            st.markdown(f"**{f}** · {len(df)} rows")
            st.dataframe(df.head(50), use_container_width=True, hide_index=True)
        elif p.suffix.lower() == ".json":
            with st.expander(f"📄 {f}"):
                st.code(p.read_text()[:4000], language="json")
        st.download_button(f"⬇ Download {p.name}", p.read_bytes(), file_name=p.name, key=f"dl_{key}_{i}")


def render_assistant(res: dict, idx: int) -> None:
    label, color = PATH_LABELS.get(res.get("path"), (res.get("path"), "#6b7280"))
    status = res.get("status")
    st.markdown(res["answer"])
    if res.get("files"):
        render_files(res["files"], key=str(idx))
    tools = ", ".join(dict.fromkeys(res.get("tools_used", []))) or "none"
    st.markdown(
        f'<span class="badge" style="background:{color}">{label}</span>'
        f'<span class="badge" style="background:{"#16a34a" if status == "success" else "#ea580c"}">{status}</span>'
        f'<span class="meta">tools: {tools} · tokens: {res["tokens"]["total"]} · '
        f'{res["time_ms"] / 1000:.1f}s</span>', unsafe_allow_html=True)
    with st.expander("🔍 How the agent worked"):
        if res.get("knowledge_used"):
            st.markdown("**Retrieved from the vector DB**")
            st.dataframe(pd.DataFrame(res["knowledge_used"]), hide_index=True, use_container_width=True)
        else:
            st.markdown("**Retrieved from the vector DB:** nothing above the threshold → direct LLM")
        st.markdown("**Steps**")
        for n, s in enumerate(res.get("steps", []), start=1):
            if s["type"] == "retrieve":
                st.markdown(f"{n}. 🧭 Route: `{s['path']}` · tools given to the LLM: "
                            f"{', '.join(f'`{t}`' for t in s['tools_available']) or 'none'}")
            elif s["type"] == "llm":
                calls = s.get("tool_calls")
                if calls:
                    st.markdown(f"{n}. 🧠 LLM decided to call: " + ", ".join(f"`{c['tool']}`" for c in calls))
                else:
                    st.markdown(f"{n}. 🧠 LLM: {s.get('note', '')}")
            elif s["type"] == "tool":
                icon = "✅" if s.get("ok") else "⚠️"
                st.markdown(f"{n}. {icon} Tool `{s['tool']}` ({s.get('ms', 0)} ms)")
                st.code(f"args: {s['args']}\n\n{s.get('result_preview', '')}", language=None)


with tab_chat:
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
            if m["role"] == "user":
                st.markdown(m["content"].split("\n\n[Attached file:")[0])
                if m.get("file"):
                    st.caption(f"📎 {m['file']}")
            else:
                render_assistant(m["res"], i)

    last = st.session_state.messages[-1] if st.session_state.messages else None
    if last and last["role"] == "assistant" and last["res"].get("needs_confirmation"):
        st.warning("The agent is waiting for your confirmation.")
        c1, c2, _ = st.columns([1, 1, 4])
        if c1.button("✅ Approve", type="primary"):
            send("yes")
            st.rerun()
        if c2.button("❌ Reject"):
            send("no")
            st.rerun()

    upload = st.file_uploader("Attach a file to your next message (CSV, Excel, PDF, image)",
                              type=["csv", "xlsx", "pdf", "docx", "txt", "png", "jpg", "jpeg"])
    prompt = st.chat_input("Give the agent a command...")
    if st.session_state.queued:
        prompt, st.session_state.queued = st.session_state.queued, None
    if prompt:
        send(prompt, upload)
        st.rerun()

# -------------------------------------------------------- knowledge tab
with tab_kb:
    st.markdown("### Add knowledge to the vector DB")
    st.caption("Everything you add is embedded with the embedding model and stored in ChromaDB. "
               "The agent searches it before every command.")
    c1, c2, c3 = st.columns(3)

    def post_knowledge(data: dict, files=None) -> None:
        try:
            r = requests.post(f"{api()}/knowledge", data=data, files=files, timeout=300)
            if r.ok:
                st.success(f"Added {r.json()['items_added']} item(s). Totals: {r.json()['totals']}")
                get_health.clear()
            else:
                st.error(r.json().get("detail", r.text))
        except requests.RequestException as e:
            st.error(f"API error: {e}")

    with c1, st.form("chat_form", clear_on_submit=True):
        st.markdown("#### 💬 Chat example")
        st.caption("A question users often ask + the ideal answer. Very similar questions get a fast answer.")
        q = st.text_input("Question")
        a = st.text_area("Answer", height=150)
        if st.form_submit_button("Add chat example", use_container_width=True):
            post_knowledge({"type": "chat", "content": q, "answer": a})

    with c2, st.form("task_form", clear_on_submit=True):
        st.markdown("#### 🧩 Task description")
        st.caption("How to do one type of task. Used as a ready-made plan, with only these tools.")
        title = st.text_input("Task title", placeholder="Compare prices on two websites")
        steps = st.text_area("Steps", height=110, placeholder="1. ...\n2. ...")
        tools = st.multiselect("Tools", TOOL_NAMES)
        examples = st.text_area("Example commands (one per line)", height=80)
        if st.form_submit_button("Add task description", use_container_width=True):
            post_knowledge({"type": "task", "title": title, "content": steps, "tools": ",".join(tools),
                            "examples": examples})

    with c3, st.form("doc_form", clear_on_submit=True):
        st.markdown("#### 📄 Document")
        st.caption("PDF, DOCX, TXT or MD. Split into chunks and used to answer questions with sources.")
        doc = st.file_uploader("File", type=["pdf", "docx", "txt", "md"])
        doc_title = st.text_input("Title (optional)")
        if st.form_submit_button("Add document", use_container_width=True):
            if doc:
                post_knowledge({"type": "document", "title": doc_title or doc.name},
                               files={"file": (doc.name, doc.getvalue())})
            else:
                st.error("Choose a file first.")

# -------------------------------------------------------- dashboard tab
with tab_dash:
    try:
        stats = requests.get(f"{api()}/stats", timeout=10).json()
    except requests.RequestException:
        stats = None
    if not stats:
        st.info("Start the API to see statistics.")
    else:
        st.markdown("### Agent performance")
        m = st.columns(4)
        m[0].metric("Total tasks", stats["total_tasks"])
        m[1].metric("Successful", stats["successful"], f"{stats['success_rate']}%")
        m[2].metric("Failed / stuck", stats["failed"])
        m[3].metric("Waiting confirmation", stats["waiting_confirmation"])
        m = st.columns(4)
        m[0].metric("Total tokens", f"{stats['total_tokens']:,}")
        m[1].metric("Avg tokens / task", f"{stats['avg_tokens_per_task']:,}")
        m[2].metric("Avg time / task", f"{stats['avg_time_ms'] / 1000:.1f}s")
        m[3].metric("RAG hit rate", f"{stats['rag_hit_rate']}%")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Path performance (does RAG help?)")
            if stats["path_performance"]:
                perf = pd.DataFrame(stats["path_performance"]).T.reset_index().rename(columns={"index": "path"})
                st.dataframe(perf, hide_index=True, use_container_width=True)
                st.bar_chart(perf.set_index("path")["avg_time_ms"], horizontal=True)
                st.caption("Compare 'playbook' and 'quick' with 'direct' to show the effect of the vector DB.")
        with c2:
            st.markdown("#### Most used tools")
            if stats["top_tools"]:
                st.bar_chart(pd.Series(stats["top_tools"]).sort_values(), horizontal=True)
        st.markdown("#### Recent tasks")
        if stats["recent_tasks"]:
            rec = pd.DataFrame(stats["recent_tasks"])
            rec["created_at"] = pd.to_datetime(rec["created_at"], unit="s").dt.strftime("%d %b %H:%M")
            rec["tools"] = rec["tools"].apply(lambda t: ", ".join(dict.fromkeys(t)))
            st.dataframe(rec, hide_index=True, use_container_width=True)
