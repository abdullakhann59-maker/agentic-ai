"""Tool registry.

Tools are organised in GROUPS. Each group has a description that is embedded in the
vector DB ("tools" collection). On the direct path the agent only receives the tools of
the most relevant groups, because small local models choose much better from 5-10 tools
than from 30.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from app.tools.browser_tools import browser_click, browser_find, browser_open, browser_screenshot, browser_type
from app.tools.data import chart_make, data_analyze, data_describe
from app.tools.documents import doc_read, invoice_extract, ocr_image
from app.tools.email_tools import email_draft, email_send
from app.tools.memory_tools import memory_save, memory_search
from app.tools.scraper_tools import (scrape_analyze, scrape_api, scrape_embedded, scrape_iframes, scrape_page,
                                     scrape_save, scrape_structure)
from app.tools.utility import calculator
from app.tools.web import web_read, web_search

TOOL_GROUPS: dict[str, dict] = {
    "web_research": {
        "description": "search the internet, look up current information, news, facts, read a web page or article",
        "tools": [web_search, web_read],
    },
    "browser": {
        "description": "open a website in the browser, click buttons and links, fill forms, type into fields, "
                       "navigate pages, take a screenshot",
        "tools": [browser_open, browser_find, browser_click, browser_type, browser_screenshot],
    },
    "scraper": {
        "description": "scrape or extract data from a website url into a file: products, prices, tables, lists; "
                       "detect dynamic pages, APIs and iframes; save CSV JSON Excel",
        "tools": [scrape_analyze, scrape_api, scrape_embedded, scrape_iframes, scrape_page, scrape_structure,
                  scrape_save],
    },
    "documents": {
        "description": "read a pdf, word docx or text document file, OCR an image or scan, extract invoice "
                       "fields like invoice number, GST, total",
        "tools": [doc_read, ocr_image, invoice_extract],
    },
    "data": {
        "description": "analyze a csv or excel spreadsheet data file: totals, averages, top rows, group by, "
                       "filter; make a bar line or pie chart or graph",
        "tools": [data_describe, data_analyze, chart_make],
    },
    "email": {
        "description": "write, draft or send an email message to someone",
        "tools": [email_draft, email_send],
    },
    "memory": {
        "description": "remember a fact or preference for later, recall what I told you before",
        "tools": [memory_save, memory_search],
    },
    "utility": {
        "description": "calculate math, arithmetic, percentages, numbers",
        "tools": [calculator],
    },
}

ALL_TOOLS: dict[str, BaseTool] = {t.name: t for g in TOOL_GROUPS.values() for t in g["tools"]}
ALWAYS_INCLUDED = ["calculator"]


def tools_by_names(names: list[str]) -> list[BaseTool]:
    out = [ALL_TOOLS[n] for n in names if n in ALL_TOOLS]
    for n in ALWAYS_INCLUDED:
        if ALL_TOOLS[n] not in out:
            out.append(ALL_TOOLS[n])
    return out


def tools_for_groups(groups: list[str]) -> list[BaseTool]:
    names = [t.name for g in groups for t in TOOL_GROUPS[g]["tools"]]
    return tools_by_names(names)


def group_descriptions() -> dict[str, str]:
    return {g: f"{g}: {v['description']}" for g, v in TOOL_GROUPS.items()}
