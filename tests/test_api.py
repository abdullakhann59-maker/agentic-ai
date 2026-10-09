"""API tests for the 4 endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api import app


def test_four_endpoints(fake_llm):
    with TestClient(app) as client:
        h = client.get("/health").json()
        assert h["ollama"] == "unreachable" and h["status"] == "degraded"   # no Ollama in tests
        assert h["vector_db"]["status"] == "ok"

        r = client.post("/knowledge", data={"type": "chat", "content": "who made you",
                                            "answer": "I was built as a portfolio project."})
        assert r.status_code == 200 and r.json()["totals"]["chat"] == 1
        r = client.post("/knowledge", data={"type": "task", "title": "Make a chart",
                                            "content": "1. data_describe 2. chart_make",
                                            "tools": "data_describe,chart_make", "examples": "plot revenue"})
        assert r.json()["totals"]["task"] == 1
        r = client.post("/knowledge", data={"type": "document", "title": "notes"},
                        files={"file": ("notes.txt", b"The office opens at 9 am and closes at 6 pm every weekday.")})
        assert r.json()["items_added"] == 1
        assert client.post("/knowledge", data={"type": "chat"}).status_code == 400

        fake_llm.replies = [AIMessage(content="I was built as a portfolio project.")]
        res = client.post("/chat", data={"message": "who made you", "session_id": "api"}).json()
        assert res["path"] == "quick" and res["status"] == "success"

        fake_llm.replies = [AIMessage(content="Got the file.")]
        res = client.post("/chat", data={"message": "here is a file"},
                          files={"file": ("my data.csv", b"a,b\n1,2\n")}).json()
        assert res["status"] == "success"

        st = client.get("/stats").json()
        assert st["total_tasks"] >= 2 and st["total_tokens"] > 0
