"""Browser automation tools (Selenium). The LLM sees numbered elements, never raw HTML."""
from __future__ import annotations

import time
from pathlib import Path

from langchain_core.tools import tool

from app.browser import browser, domain_allowed
from app.config import get_settings
from app.context import ctx


@tool
def browser_open(url: str) -> str:
    """Open a URL in the browser and return the page title and visible text (trimmed).
    Only domains in ALLOWED_DOMAINS can be opened."""
    if not domain_allowed(url):
        return f"Error: {url} is not in the allowed domains. Ask the user to add it to ALLOWED_DOMAINS."
    with browser.lock:
        browser.open(url)
        d = browser.driver()
        text = d.find_element("tag name", "body").text
        lowered = d.page_source.lower()
        if "captcha" in lowered or d.find_elements("css selector", "input[type=password]"):
            note = "\nNOTE: this page has a CAPTCHA or login form. Stop and ask the user; do not try to bypass it."
        else:
            note = ""
        return f"Opened: {d.title}\n{text[: get_settings().tool_output_chars]}{note}"


@tool
def browser_find(description: str = "") -> str:
    """List clickable/typeable elements on the current page as numbered items, e.g. [3] button "Search".
    Optionally filter by words in `description`. Use the number with browser_click or browser_type."""
    with browser.lock:
        d = browser.driver()
        els = d.find_elements("css selector", "a, button, input, select, textarea, [role=button]")
        words = [w for w in description.lower().split() if len(w) > 2]
        lines = []
        browser.elements = {}
        n = 0
        for el in els:
            try:
                if not el.is_displayed():
                    continue
                label = (el.text or el.get_attribute("aria-label") or el.get_attribute("placeholder")
                         or el.get_attribute("name") or el.get_attribute("value") or "").strip()[:60]
                kind = el.tag_name if el.tag_name != "input" else f"input[{el.get_attribute('type') or 'text'}]"
            except Exception:
                continue
            if words and not any(w in label.lower() for w in words):
                continue
            n += 1
            browser.elements[n] = el
            lines.append(f"[{n}] {kind} \"{label}\"")
            if n >= 40:
                break
        return "\n".join(lines) or "No matching elements found."


@tool
def browser_click(element_id: int) -> str:
    """Click an element by the number shown by browser_find."""
    with browser.lock:
        el = browser.elements.get(int(element_id))
        if el is None:
            return "Error: unknown element id. Call browser_find first."
        el.click()
        browser.wait_idle(idle_seconds=0.8, max_wait=6)
        d = browser.driver()
        return f"Clicked. Page is now: {d.title}\n{d.find_element('tag name', 'body').text[:1500]}"


@tool
def browser_type(element_id: int, text: str, press_enter: bool = False) -> str:
    """Type text into an input element (number from browser_find). Never type passwords."""
    from selenium.webdriver.common.keys import Keys

    with browser.lock:
        el = browser.elements.get(int(element_id))
        if el is None:
            return "Error: unknown element id. Call browser_find first."
        if (el.get_attribute("type") or "").lower() == "password":
            return "Error: typing passwords is not allowed. Ask the user to log in themselves."
        el.clear()
        el.send_keys(text + (Keys.ENTER if press_enter else ""))
        if press_enter:
            browser.wait_idle(idle_seconds=0.8, max_wait=6)
        return "Typed the text." + (" Pressed Enter." if press_enter else "")


@tool
def browser_screenshot() -> str:
    """Save a screenshot of the current page and return the file path."""
    folder = Path(get_settings().workspace_dir) / "screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"screenshot_{int(time.time())}.png"
    with browser.lock:
        browser.driver().save_screenshot(str(path))
    rel = str(path.relative_to(get_settings().workspace_dir))
    ctx().add_file(rel)
    return f"Screenshot saved: {rel}"
