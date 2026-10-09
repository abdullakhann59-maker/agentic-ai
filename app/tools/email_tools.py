"""Email tools. Drafting is free; sending always needs the user's confirmation."""
from __future__ import annotations

import json
import re
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path

from langchain_core.tools import tool

from app.config import get_settings
from app.context import ctx
from app.tools.confirm import register, request_confirmation

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _folder() -> Path:
    f = Path(get_settings().workspace_dir) / "emails"
    f.mkdir(parents=True, exist_ok=True)
    return f


@tool
def email_draft(to: str, subject: str, body: str) -> str:
    """Write an email draft and save it. Returns a draft_id. This does NOT send anything."""
    if not EMAIL_RE.match(to.strip()):
        return f"Error: '{to}' is not a valid email address."
    draft_id = f"draft_{int(time.time() * 1000)}"
    (_folder() / f"{draft_id}.json").write_text(json.dumps({"to": to, "subject": subject, "body": body}, indent=2))
    ctx().add_file(f"emails/{draft_id}.json")
    return f"Draft saved with draft_id={draft_id}.\nTo: {to}\nSubject: {subject}\n\n{body}"


@tool
def email_send(draft_id: str) -> str:
    """Send a saved email draft. Needs the user's confirmation before anything is sent."""
    path = _folder() / f"{draft_id}.json"
    if not path.exists():
        return f"Error: draft {draft_id} not found. Create it with email_draft first."
    d = json.loads(path.read_text())
    return request_confirmation("email_send", {"draft_id": draft_id},
                                f"Send the email '{d['subject']}' to {d['to']}.")


@register("email_send")
def _send(draft_id: str) -> str:
    s = get_settings()
    d = json.loads((_folder() / f"{draft_id}.json").read_text())
    if s.email_demo_mode:
        with open(_folder() / "outbox.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(d | {"sent_at": time.time()}) + "\n")
        return f"Demo mode: email to {d['to']} logged in emails/outbox.log (not really sent)."
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = s.smtp_from or s.smtp_user, d["to"], d["subject"]
    msg.set_content(d["body"])
    with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(s.smtp_user, s.smtp_password)
        smtp.send_message(msg)
    return f"Email sent to {d['to']}."
