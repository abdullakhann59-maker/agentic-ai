"""SQLite storage: task stats (for /stats), chat history and pending confirmations."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path

from app.config import get_settings

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    session_id TEXT,
    message TEXT,
    path TEXT,
    status TEXT,
    tools_used TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    time_ms INTEGER DEFAULT 0,
    rag_hit INTEGER DEFAULT 0,
    error TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_actions (
    session_id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    path = Path(get_settings().sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(SCHEMA)


# ---------------- tasks / stats ----------------
def record_task(*, session_id: str, message: str, path: str, status: str, tools_used: list[str],
                input_tokens: int, output_tokens: int, time_ms: int, rag_hit: bool,
                error: str | None = None) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO tasks (created_at, session_id, message, path, status, tools_used, input_tokens,"
            " output_tokens, time_ms, rag_hit, error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), session_id, message[:500], path, status, json.dumps(tools_used),
             input_tokens, output_tokens, time_ms, int(rag_hit), error),
        )


def get_stats(recent: int = 15) -> dict:
    with _lock, _connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()]
    total = len(rows)
    success = sum(1 for r in rows if r["status"] == "success")
    failed = sum(1 for r in rows if r["status"] in ("failed", "stuck"))
    waiting = sum(1 for r in rows if r["status"] == "waiting_confirmation")
    inp = sum(r["input_tokens"] for r in rows)
    out = sum(r["output_tokens"] for r in rows)
    tools = Counter()
    for r in rows:
        tools.update(json.loads(r["tools_used"] or "[]"))
    paths = Counter(r["path"] for r in rows)
    path_perf = {}
    for p in paths:
        sub = [r for r in rows if r["path"] == p]
        finished = [r for r in sub if r["status"] != "waiting_confirmation"] or sub
        path_perf[p] = {
            "count": len(sub),
            "success_rate": round(100 * sum(r["status"] == "success" for r in finished) / len(finished), 1),
            "avg_time_ms": int(sum(r["time_ms"] for r in sub) / len(sub)),
            "avg_tokens": int(sum(r["input_tokens"] + r["output_tokens"] for r in sub) / len(sub)),
        }
    return {
        "total_tasks": total,
        "successful": success,
        "failed": failed,
        "waiting_confirmation": waiting,
        # tasks still waiting for the user's "yes" are not finished, so they are left out of the rate
        "success_rate": round(100 * success / (total - waiting), 1) if total - waiting else 0.0,
        "total_tokens": inp + out,
        "input_tokens": inp,
        "output_tokens": out,
        "avg_tokens_per_task": int((inp + out) / total) if total else 0,
        "avg_time_ms": int(sum(r["time_ms"] for r in rows) / total) if total else 0,
        "rag_hit_rate": round(100 * sum(r["rag_hit"] for r in rows) / total, 1) if total else 0.0,
        "path_counts": dict(paths),
        "path_performance": path_perf,
        "top_tools": dict(tools.most_common(10)),
        "recent_tasks": [
            {k: r[k] for k in ("created_at", "message", "path", "status", "time_ms")}
            | {"tokens": r["input_tokens"] + r["output_tokens"], "tools": json.loads(r["tools_used"] or "[]")}
            for r in rows[:recent]
        ],
    }


# ---------------- chat history ----------------
def add_message(session_id: str, role: str, content: str) -> None:
    with _lock, _connect() as conn:
        conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                     (session_id, role, content, time.time()))


def get_history(session_id: str, limit: int) -> list[dict]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]


# ---------------- pending confirmations ----------------
def set_pending(session_id: str, action: dict) -> None:
    with _lock, _connect() as conn:
        conn.execute("INSERT OR REPLACE INTO pending_actions VALUES (?,?,?)",
                     (session_id, json.dumps(action), time.time()))


def pop_pending(session_id: str) -> dict | None:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT action FROM pending_actions WHERE session_id=?", (session_id,)).fetchone()
        conn.execute("DELETE FROM pending_actions WHERE session_id=?", (session_id,))
    return json.loads(row["action"]) if row else None


def has_pending(session_id: str) -> bool:
    with _lock, _connect() as conn:
        return conn.execute("SELECT 1 FROM pending_actions WHERE session_id=?", (session_id,)).fetchone() is not None


def ping() -> bool:
    with _lock, _connect() as conn:
        conn.execute("SELECT 1")
    return True
