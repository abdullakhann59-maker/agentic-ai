"""FastAPI app with the 4 endpoints.

  POST /chat       talk to the agent (optional file attachment)
  POST /knowledge  add chat examples, task descriptions or documents to the vector DB
  GET  /health     status of Ollama, the vector DB, SQLite and Selenium
  GET  /stats      task counts, success rate, tokens, path and tool usage
"""
from __future__ import annotations

import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import storage
from app.agent import ensure_tool_index, run_chat
from app.browser import selenium_ready
from app.config import get_settings
from app.knowledge import get_kb


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    try:
        ensure_tool_index()          # needs the embedding model; skip quietly if Ollama is down
    except Exception as e:  # noqa: BLE001
        print(f"[startup] Tool index not built yet (is Ollama running?): {e}")
    yield
    from app.browser import browser
    browser.close()


app = FastAPI(title="Nexus Agent", version="1.0", lifespan=lifespan,
              description="Agentic AI with RAG, LangGraph, ChromaDB and Selenium tools.")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name) or "upload"


@app.post("/chat")
async def chat(message: str = Form(...), session_id: str = Form("default"),
               file: UploadFile | None = File(None)):
    """Send a command to the agent. Attach a file (CSV, PDF, invoice, image) if the task needs one."""
    if not message.strip():
        raise HTTPException(400, "message is empty")
    if file is not None and file.filename:
        folder = Path(get_settings().workspace_dir) / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / _safe_name(file.filename)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        message += f"\n\n[Attached file: uploads/{dest.name}]"
    ensure_index_once()
    return await run_in_threadpool(run_chat, message, session_id)


_index_ready = False


def ensure_index_once() -> None:
    global _index_ready
    if not _index_ready:
        ensure_tool_index()
        _index_ready = True


@app.post("/knowledge")
async def knowledge(
    type: Literal["chat", "task", "document"] = Form(...),
    title: str = Form(""),
    content: str = Form(""),
    answer: str = Form(""),
    tools: str = Form(""),
    examples: str = Form(""),
    file: UploadFile | None = File(None),
):
    """Add knowledge to the vector DB.

    - chat:     content = the question, answer = the answer
    - task:     title, content = steps/description, tools = comma-separated tool names,
                examples = example commands, one per line
    - document: upload a file (PDF, DOCX, TXT, MD) or send text in content (with a title)
    """
    kb = get_kb()
    if type == "chat":
        if not content.strip() or not answer.strip():
            raise HTTPException(400, "chat needs content (question) and answer")
        added = await run_in_threadpool(kb.add_chat, content.strip(), answer.strip())
    elif type == "task":
        if not title.strip() or not content.strip():
            raise HTTPException(400, "task needs a title and content (the steps)")
        tool_list = [t.strip() for t in tools.split(",") if t.strip()]
        ex = [e.strip() for e in examples.splitlines() if e.strip()]
        added = await run_in_threadpool(kb.add_task, title.strip(), content.strip(), tool_list, ex)
    else:
        if file is not None and file.filename:
            folder = Path(get_settings().data_dir) / "knowledge_files"
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / f"{uuid.uuid4().hex[:8]}_{_safe_name(file.filename)}"
            with dest.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            try:
                added = await run_in_threadpool(kb.add_document, dest, title or file.filename)
            except ValueError as e:
                raise HTTPException(400, str(e))
        elif content.strip():
            added = await run_in_threadpool(kb.add_text_document, title or "note", content)
        else:
            raise HTTPException(400, "document needs a file or content")
    return {"type": type, "items_added": added, "totals": kb.counts(),
            "note": "0 items added means this document was already stored." if added == 0 else ""}


@app.get("/health")
def health():
    s = get_settings()
    out: dict = {"status": "ok"}
    try:
        tags = requests.get(f"{s.ollama_base_url}/api/tags", timeout=3).json()
        names = [m["name"] for m in tags.get("models", [])]
        def has(model: str) -> bool:
            return any(n == model or n.startswith(model + ":") for n in names)
        out["ollama"] = "ok"
        out["llm_model"] = {"name": s.llm_model, "available": has(s.llm_model)}
        out["embedding_model"] = {"name": s.embed_model, "available": has(s.embed_model)}
    except Exception:
        out["ollama"] = "unreachable"
    try:
        out["vector_db"] = {"status": "ok", "items": get_kb().counts()}
    except Exception as e:
        out["vector_db"] = {"status": "error", "detail": str(e)[:200]}
    try:
        storage.ping()
        out["stats_db"] = "ok"
    except Exception:
        out["stats_db"] = "error"
    out["selenium"] = "ok" if selenium_ready() else "not installed"
    out["workspace"] = str(s.workspace_dir)
    if out.get("ollama") != "ok" or out["vector_db"].get("status") != "ok" or out["stats_db"] != "ok":
        out["status"] = "degraded"
    return out


@app.get("/stats")
def stats():
    return storage.get_stats()
