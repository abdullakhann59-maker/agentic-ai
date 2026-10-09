"""Agent graph tests with a scripted LLM: each path, tools, confirmation, loop limits, stats."""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from app import storage
from app.agent import run_chat
from app.config import get_settings
from app.knowledge import get_kb
from tests.fakes import tool_call


def test_quick_path_uses_chat_example(fake_llm):
    get_kb().add_chat("what can you do", "I can scrape, research and analyze data.")
    fake_llm.replies = [AIMessage(content="I can scrape websites, research and analyze data.")]
    res = run_chat("what can you do?", "s_quick")
    assert res["path"] == "quick"
    assert res["status"] == "success"
    assert res["tools_used"] == []
    assert res["tokens"]["total"] == 120
    # the stored answer was given to the LLM as a reference
    assert "I can scrape, research" in fake_llm.calls[0][0].content


def test_playbook_path_follows_task_and_limits_tools(fake_llm):
    get_kb().add_task("Analyze a data file", "1. data_describe 2. data_analyze", ["data_describe", "data_analyze"],
                      ["total revenue by region in sales_data.csv", "analyze sales csv file"])
    fake_llm.replies = [
        tool_call("data_describe", {"file_path": "uploads/sales_data.csv"}, "a"),
        tool_call("data_analyze", {"file_path": "uploads/sales_data.csv", "operation": "group_sum",
                                   "column": "revenue", "by": "region"}, "b"),
        AIMessage(content="Revenue by region: ..."),
    ]
    res = run_chat("total revenue by region in sales_data.csv", "s_play")
    assert res["path"] == "playbook"
    assert res["tools_used"] == ["data_describe", "data_analyze"]
    assert set(fake_llm.bound_tools[0]) == {"data_describe", "data_analyze", "calculator"}
    tool_steps = [s for s in res["steps"] if s["type"] == "tool"]
    assert all(s["ok"] for s in tool_steps)
    assert "North" in tool_steps[1]["result_preview"]
    assert "Plan to follow" in fake_llm.calls[0][0].content


def test_direct_path_picks_tool_groups_from_vector_db(fake_llm):
    fake_llm.replies = [tool_call("calculator", {"expression": "18 * 1.18"}), AIMessage(content="21.24")]
    res = run_chat("calculate 18 * 1.18 math", "s_direct")
    assert res["path"] == "direct"
    assert "calculator" in fake_llm.bound_tools[0]
    assert len(fake_llm.bound_tools[0]) < 15       # not all 25 tools
    assert res["answer"] == "21.24"


def test_document_rag_adds_sources(fake_llm):
    get_kb().add_text_document("handbook.md", "Employees get 24 days of paid leave per year in the leave policy.")
    fake_llm.replies = [AIMessage(content="You get 24 days of paid leave.")]
    res = run_chat("how many days of paid leave do employees get per year leave policy", "s_doc")
    assert any(k["type"] == "document" for k in res["knowledge_used"])
    assert "handbook.md" in res["answer"]          # check node appended the source


def test_email_needs_confirmation_then_sends(fake_llm):
    fake_llm.replies = [
        tool_call("email_draft", {"to": "priya@example.com", "subject": "Hi", "body": "Hello"}, "d"),
        lambda msgs: tool_call("email_send", {"draft_id": _draft_id(msgs)}, "s"),
    ]
    res = run_chat("send an email to priya@example.com saying hello", "s_mail")
    assert res["status"] == "waiting_confirmation"
    assert res["needs_confirmation"] is True
    outbox = Path(get_settings().workspace_dir) / "emails" / "outbox.log"
    assert not outbox.exists()
    res2 = run_chat("yes", "s_mail")
    assert res2["path"] == "confirmation" and res2["status"] == "success"
    assert outbox.exists() and "priya@example.com" in outbox.read_text()


def test_cancel_confirmation(fake_llm):
    fake_llm.replies = [
        tool_call("email_draft", {"to": "a@example.com", "subject": "x", "body": "y"}, "d"),
        lambda msgs: tool_call("email_send", {"draft_id": _draft_id(msgs)}, "s"),
    ]
    run_chat("email a@example.com", "s_cancel")
    res = run_chat("no", "s_cancel")
    assert "Cancelled" in res["answer"]
    assert not storage.has_pending("s_cancel")


def test_loop_detection_stops_agent(fake_llm):
    fake_llm.replies = [tool_call("calculator", {"expression": "1+1"}, f"c{i}") for i in range(10)]
    res = run_chat("calculate 1+1 math", "s_loop")
    assert res["status"] == "stuck"


def test_unknown_tool_is_reported_not_crashing(fake_llm):
    fake_llm.replies = [tool_call("delete_everything", {}), AIMessage(content="I can't do that.")]
    res = run_chat("calculate something math", "s_unknown")
    assert res["status"] == "success"
    tool_step = next(s for s in res["steps"] if s["type"] == "tool")
    assert tool_step["ok"] is False and "not available" in tool_step["result_preview"]


def test_stats_counts_tasks_and_tokens(fake_llm):
    before = storage.get_stats()["total_tasks"]
    fake_llm.replies = [AIMessage(content="hello")]
    run_chat("hello there friend", "s_stats")
    st = storage.get_stats()
    assert st["total_tasks"] == before + 1
    assert st["total_tokens"] > 0
    assert "direct" in st["path_counts"] or "quick" in st["path_counts"]


def _draft_id(msgs) -> str:
    import re
    for m in reversed(msgs):
        found = re.search(r"draft_id=(draft_\d+)", str(m.content))
        if found:
            return found.group(1)
    raise AssertionError("no draft id")
