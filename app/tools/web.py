"""Web research tools: search the web and read a page as clean text."""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from app.config import get_settings


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet and return titles, links and snippets.
    Use for current facts, news, or finding pages. Do NOT use for files the user uploaded."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=max(1, min(max_results, 10)))
    except Exception as e:  # network blocked, rate limit, etc.
        return f"Error: web search failed ({type(e).__name__}: {e}). Try again later or use web_read on a known URL."
    if not results:
        return "No results found."
    return "\n\n".join(f"[{i + 1}] {r.get('title')}\n{r.get('href')}\n{r.get('body')}" for i, r in enumerate(results))


@tool
def web_read(url: str) -> str:
    """Download a web page and return its main text (trimmed). Use to read an article or page.
    Do NOT use to extract tables/products into a file (use the scraper tools for that)."""
    s = get_settings()
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": s.user_agent})
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Error: could not read {url}: {e}"
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    # hidden elements are a common place for prompt-injection text: drop them
    for tag in soup.select('[style*="display:none"], [style*="display: none"], [hidden]'):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else url
    text = " ".join(soup.get_text(" ", strip=True).split())
    return f"Title: {title}\nURL: {url}\n\n{text[: s.tool_output_chars]}"
