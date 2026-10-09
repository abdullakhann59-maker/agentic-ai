"""The agent: a LangGraph state machine.

    retrieve ──┬── quick_answer ─────────────────────────────┐
               │                                             ▼
               └── agent ⇄ tools  (loop with limits) ── check ── finalize

retrieve  : embed the command once, search the vector DB, choose the path
            - "quick"    : a stored chat example matches very closely -> one short LLM call, no tools
            - "playbook" : a stored task description matches -> use it as the plan, only its tools
            - "direct"   : nothing relevant -> LLM works on its own with the most relevant tool groups
agent     : LLM decides the next tool call (or gives the final answer)
tools     : runs the tool calls safely (errors become text, loops are detected, risky actions paused)
check     : rule-based self-check of the final answer
"""
from __future__ import annotations

import json
import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import prompts, storage
from app.config import get_settings
from app.context import RunContext, ctx, reset_context, set_context
from app.knowledge import get_kb
from app.llm import get_chat_model, token_usage
from app.tools import ALL_TOOLS, TOOL_GROUPS, group_descriptions, tools_by_names, tools_for_groups
from app.tools import confirm


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_message: str
    search_text: str          # what to search the vector DB with (defaults to user_message)
    session_id: str
    path: str
    tool_names: list[str]
    knowledge_used: list[dict]
    steps: Annotated[list[dict], operator.add]
    step_count: int
    consecutive_errors: int
    call_counts: dict
    status: str
    answer: str
    started: float
    checked: bool
    rag_hit: bool
    doc_sources: list[str]


# ------------------------------------------------------------------ nodes
def retrieve(state: AgentState) -> dict:
    s = get_settings()
    kb = get_kb()
    r = kb.retrieve(state.get("search_text") or state["user_message"])
    used: list[dict] = []
    system = prompts.SYSTEM_BASE

    chat = r.chat[0] if r.chat and r.chat[0].score >= s.chat_threshold else None
    task = r.tasks[0] if r.tasks and r.tasks[0].score >= s.task_threshold else None

    if chat and (task is None or chat.score >= task.score):
        path, tool_names = "quick", []
        system += prompts.QUICK_SECTION.format(question=chat.meta.get("question", chat.text), answer=chat.meta["answer"])
        used.append({"type": "chat", "title": chat.meta.get("question", chat.text), "score": chat.score})
    elif task:
        path = "playbook"
        tool_names = [t.strip() for t in task.meta.get("tools", "").split(",") if t.strip() in ALL_TOOLS]
        if not tool_names:  # playbook without valid tools: fall back to group selection
            tool_names = [t.name for t in tools_for_groups([g for g, _ in kb.select_tool_groups(
                r.vector, s.tool_group_top_k)])]
        system += prompts.PLAYBOOK_SECTION.format(title=task.meta["title"], description=task.meta["description"])
        used.append({"type": "task", "title": task.meta["title"], "score": task.score})
    else:
        path = "direct"
        groups = [g for g, _ in kb.select_tool_groups(r.vector, s.tool_group_top_k)] or list(TOOL_GROUPS)
        tool_names = [t.name for t in tools_for_groups(groups)]

    docs = [h for h in r.docs if h.score >= s.doc_threshold]
    sources = []
    if docs:
        chunks = "\n\n".join(f"[{h.meta.get('source')}, page {h.meta.get('page')}]\n{h.text}" for h in docs)
        system += prompts.KNOWLEDGE_SECTION.format(chunks=chunks)
        for h in docs:
            used.append({"type": "document", "title": f"{h.meta.get('source')} p.{h.meta.get('page')}",
                         "score": h.score})
            sources.append(str(h.meta.get("source")))
    mem = [h for h in r.memory if h.score >= s.memory_threshold]
    if mem:
        system += prompts.MEMORY_SECTION.format(facts="\n".join(f"- {h.text}" for h in mem))
        used += [{"type": "memory", "title": h.text[:60], "score": h.score} for h in mem]

    history = []
    for m in storage.get_history(state["session_id"], s.history_messages):
        history.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))

    return {
        "messages": [SystemMessage(system), *history, HumanMessage(state["user_message"])],
        "path": path,
        "tool_names": tool_names,
        "knowledge_used": used,
        "rag_hit": bool(used),
        "doc_sources": list(dict.fromkeys(sources)),
        "steps": [{"type": "retrieve", "path": path, "tools_available": tool_names,
                   "knowledge": used}],
    }


def quick_answer(state: AgentState) -> dict:
    msg = get_chat_model().invoke(state["messages"])
    ctx().add_tokens(*token_usage(msg))
    return {"messages": [msg], "answer": str(msg.content), "status": "success",
            "steps": [{"type": "llm", "note": "quick answer from chat knowledge"}]}


def agent(state: AgentState) -> dict:
    s = get_settings()
    if time.time() - state["started"] > s.max_seconds:
        return {"status": "stuck", "answer": "Stopped: the task took longer than the time limit."}
    tools = tools_by_names(state["tool_names"])
    llm = get_chat_model().bind_tools(tools)
    msg = llm.invoke(state["messages"])
    ctx().add_tokens(*token_usage(msg))
    calls = [{"tool": c["name"], "args": c["args"]} for c in getattr(msg, "tool_calls", [])]
    step = {"type": "llm", "tool_calls": calls} if calls else {"type": "llm", "note": "final answer"}
    return {"messages": [msg], "step_count": state.get("step_count", 0) + 1, "steps": [step]}


def run_tools(state: AgentState) -> dict:
    s = get_settings()
    last: AIMessage = state["messages"][-1]
    allowed = set(state["tool_names"]) | {"calculator"}
    counts = dict(state.get("call_counts") or {})
    errors = state.get("consecutive_errors", 0)
    out_msgs, steps = [], []
    status = state.get("status")

    for call in last.tool_calls:
        name, args = call["name"], call["args"]
        sig = name + json.dumps(args, sort_keys=True, default=str)
        counts[sig] = counts.get(sig, 0) + 1
        t0 = time.time()
        if name not in allowed:
            result, ok = f"Error: tool '{name}' is not available. Available: {sorted(allowed)}", False
        elif counts[sig] > s.max_repeat_calls:
            result, ok, status = "Stopped: the same tool was called with the same input too many times.", False, "stuck"
        else:
            try:
                result, ok = str(ALL_TOOLS[name].invoke(args)), True
                ok = not result.startswith("Error")
            except Exception as e:  # tool crashed: give the error to the LLM so it can recover
                result, ok = f"Error: {type(e).__name__}: {e}", False
        errors = 0 if ok else errors + 1
        result = result[: s.tool_output_chars * 2]
        out_msgs.append(ToolMessage(content=f"<tool_result>\n{result}\n</tool_result>", tool_call_id=call["id"],
                                    name=name))
        steps.append({"type": "tool", "tool": name, "args": args, "ok": ok,
                      "result_preview": result[:400], "ms": int((time.time() - t0) * 1000)})
        if ctx().pending_action:
            status = "waiting_confirmation"
            break

    if errors >= s.max_consecutive_errors and status is None:
        status = "failed"
    update: dict[str, Any] = {"messages": out_msgs, "steps": steps, "call_counts": counts,
                              "consecutive_errors": errors}
    if status:
        update["status"] = status
    if status == "waiting_confirmation":
        update["answer"] = (f"I need your confirmation before I continue: {ctx().pending_action['summary']}\n"
                            "Reply **yes** to confirm or **no** to cancel.")
    elif status == "failed":
        update["answer"] = "I could not complete the task because the tools kept failing:\n" + steps[-1]["result_preview"]
    elif status == "stuck":
        update["answer"] = "I stopped because I was repeating the same step. Please rephrase or give more detail."
    return update


def check(state: AgentState) -> dict:
    """Rule-based self-check (fast, no extra LLM call)."""
    answer = str(state["messages"][-1].content or "")
    answer = re.sub(r"</?tool_result>", "", answer).strip()   # never leak the internal data markers
    if not answer and not state.get("checked"):
        return {"checked": True,
                "messages": [HumanMessage("Your last reply was empty. Give the final answer to my request now.")]}
    sources = state.get("doc_sources") or []
    if sources and not any(src.lower() in answer.lower() for src in sources):
        answer += "\n\nSources: " + ", ".join(sources)
    return {"checked": True, "answer": answer or "I could not produce an answer.", "status": "success"}


# ----------------------------------------------------------------- routing
def route_after_retrieve(state: AgentState) -> str:
    return "quick_answer" if state["path"] == "quick" else "agent"


def route_after_agent(state: AgentState) -> str:
    if state.get("status") in ("stuck", "failed"):
        return "finalize"
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        if state.get("step_count", 0) > get_settings().max_steps:
            return "limit"
        return "tools"
    return "check"


def route_after_tools(state: AgentState) -> str:
    return "finalize" if state.get("status") in ("waiting_confirmation", "stuck", "failed") else "agent"


def route_after_check(state: AgentState) -> str:
    return "finalize" if state.get("answer") else "agent"


def step_limit(state: AgentState) -> dict:
    return {"status": "stuck", "answer": "Stopped: the task needed more steps than the limit allows."}


def finalize(state: AgentState) -> dict:
    return {}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("quick_answer", quick_answer)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("check", check)
    g.add_node("limit", step_limit)
    g.add_node("finalize", finalize)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, ["quick_answer", "agent"])
    g.add_edge("quick_answer", "finalize")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "check", "limit", "finalize"])
    g.add_conditional_edges("tools", route_after_tools, ["agent", "finalize"])
    g.add_conditional_edges("check", route_after_check, ["agent", "finalize"])
    g.add_edge("limit", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_graph = None


def graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ------------------------------------------------------------- entry point
YES = re.compile(r"^\s*(yes|y|yeah|yep|confirm|confirmed|approve|approved|go ahead|ok|okay|send it|do it)\b", re.I)
NO = re.compile(r"^\s*(no|n|nope|cancel|stop|don't|dont|reject)\b", re.I)


def ensure_tool_index() -> None:
    get_kb().sync_tool_groups(group_descriptions())


def run_chat(message: str, session_id: str = "default") -> dict:
    """Handle one /chat request end to end and record it for /stats."""
    started = time.time()
    run_ctx = RunContext(session_id=session_id)
    token = set_context(run_ctx)
    result: dict[str, Any]
    try:
        pending = storage.pop_pending(session_id)
        if pending and YES.match(message):
            t0 = time.time()
            try:
                answer, status = confirm.execute(pending), "success"
            except Exception as e:
                answer, status = f"The confirmed action failed: {e}", "failed"
            step = {"type": "tool", "tool": pending["tool"], "args": pending["args"], "ok": status == "success",
                    "result_preview": answer[:400], "ms": int((time.time() - t0) * 1000)}
            result = {"answer": answer, "path": "confirmation", "status": status, "steps": [step],
                      "knowledge_used": [], "rag_hit": False}
            if status == "success" and pending.get("continue") and pending.get("request"):
                # the confirmed step was one part of a bigger task: let the agent finish the rest
                follow_up = (f"{pending['request']}\n\n[The user confirmed. {pending['tool']} already ran and "
                             f"returned:]\n{answer}\nContinue with the remaining steps. Do not call "
                             f"{pending['tool']} again.")
                state = graph().invoke(
                    {"user_message": follow_up, "search_text": pending["request"], "session_id": session_id,
                     "started": started, "steps": [step]},
                    config={"recursion_limit": get_settings().max_steps * 3 + 20})
                result.update(answer=state.get("answer") or str(state["messages"][-1].content),
                              path=state.get("path", "playbook"), status=state.get("status", "success"),
                              steps=state.get("steps", [step]), knowledge_used=state.get("knowledge_used", []),
                              rag_hit=state.get("rag_hit", False))
                if run_ctx.pending_action:
                    storage.set_pending(session_id, run_ctx.pending_action | {"request": pending["request"]})
        elif pending and NO.match(message):
            result = {"answer": "Cancelled. Nothing was done.", "path": "confirmation", "status": "success",
                      "steps": [], "knowledge_used": [], "rag_hit": False}
        else:
            state = graph().invoke(
                {"user_message": message, "session_id": session_id, "started": started, "steps": []},
                config={"recursion_limit": get_settings().max_steps * 3 + 20},
            )
            result = {"answer": state.get("answer") or str(state["messages"][-1].content),
                      "path": state.get("path", "direct"), "status": state.get("status", "success"),
                      "steps": state.get("steps", []), "knowledge_used": state.get("knowledge_used", []),
                      "rag_hit": state.get("rag_hit", False)}
            if run_ctx.pending_action:
                storage.set_pending(session_id, run_ctx.pending_action | {"request": message})
    except Exception as e:
        result = {"answer": f"Sorry, something went wrong: {type(e).__name__}: {e}", "path": "error",
                  "status": "failed", "steps": [], "knowledge_used": [], "rag_hit": False, "error": str(e)}
    finally:
        reset_context(token)

    tools_used = [s["tool"] for s in result["steps"] if s.get("type") == "tool"]
    time_ms = int((time.time() - started) * 1000)
    storage.add_message(session_id, "user", message)
    storage.add_message(session_id, "assistant", result["answer"])
    storage.record_task(session_id=session_id, message=message, path=result["path"], status=result["status"],
                        tools_used=tools_used, input_tokens=run_ctx.input_tokens,
                        output_tokens=run_ctx.output_tokens, time_ms=time_ms, rag_hit=result["rag_hit"],
                        error=result.get("error"))
    return {
        "answer": result["answer"],
        "status": result["status"],
        "path": result["path"],
        "knowledge_used": result["knowledge_used"],
        "tools_used": tools_used,
        "steps": result["steps"],
        "files": run_ctx.files,
        "needs_confirmation": result["status"] == "waiting_confirmation",
        "tokens": {"input": run_ctx.input_tokens, "output": run_ctx.output_tokens,
                   "total": run_ctx.input_tokens + run_ctx.output_tokens},
        "time_ms": time_ms,
        "prompt_version": prompts.PROMPT_VERSION,
    }
