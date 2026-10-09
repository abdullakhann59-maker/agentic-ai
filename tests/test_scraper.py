"""Scraper tests on the local demo sites (need Chrome + chromedriver)."""
from __future__ import annotations

import pytest

from app import scraper as sc
from app.tools.web import web_read
from tests.conftest import chrome_available

pytestmark = pytest.mark.skipif(not chrome_available(), reason="Chrome not available")


def test_static_site(demo_server):
    p = sc.analyze_site(f"{demo_server}/static/")
    assert p["site_type"] == "static" and p["recommended"]["strategy"] == "page"
    rows = sc.extract_page(f"{demo_server}/static/", 1)
    assert len(rows) == 8


def test_dynamic_site_finds_api(demo_server):
    p = sc.analyze_site(f"{demo_server}/dynamic/")
    assert p["site_type"] == "dynamic"
    assert p["recommended"]["strategy"] == "api"
    assert "/api/products" in p["api_candidates"][0]["url"]
    rows = sc.extract_api(p, "api_0", max_pages=5)
    assert len(rows) == 24                       # 3 pages x 8, pagination followed


def test_iframe_site_counts_and_extracts(demo_server):
    p = sc.analyze_site(f"{demo_server}/iframe/")
    assert p["iframe_count"] == 4
    kinds = {f["id"]: f["kind"] for f in p["iframes"]}
    assert kinds["2"] == "ad/tracker" and kinds["1.0"] == "content"   # nested frame found, ad skipped
    rows = sc.extract_iframes(f"{demo_server}/iframe/", p["recommended"]["iframe_ids"])
    assert {r["_frame"] for r in rows} == {"0", "1", "1.0"}


def test_embedded_json(demo_server):
    p = sc.analyze_site(f"{demo_server}/embedded/")
    assert p["recommended"]["strategy"] == "embedded"
    assert len(sc.extract_embedded(p, "emb_0")) == 8


def test_blockers(demo_server):
    assert sc.analyze_site(f"{demo_server}/captcha/")["recommended"]["strategy"] == "stop"
    assert sc.analyze_site(f"{demo_server}/private/")["allowed"] is False
    assert sc.analyze_site("https://example.com/")["allowed"] is False      # not in allowlist


def test_hidden_injection_text_is_dropped(demo_server):
    text = web_read.invoke({"url": f"{demo_server}/injection/"})
    assert "Q3 results" in text and "attacker@example.com" not in text


def test_agent_scrapes_dynamic_site_end_to_end(fake_llm, demo_server):
    """Full flow through the agent: analyze -> API (confirm) -> yes -> structure -> save CSV."""
    import json
    import re
    from pathlib import Path

    from langchain_core.messages import AIMessage

    from app.agent import run_chat
    from app.config import get_settings
    from app.knowledge import get_kb
    from tests.fakes import tool_call

    seed = json.loads((Path(__file__).parent.parent / "data/seed/task_playbooks.json").read_text())
    t = seed[0]
    get_kb().add_task(t["title"], t["description"], t["tools"], t["examples"])
    url = f"{demo_server}/dynamic/"

    def last_dataset(msgs):
        return re.findall(r'"dataset_id": "(ds_\d+)"', " ".join(str(m.content) for m in msgs))[-1]

    fake_llm.replies = [
        tool_call("scrape_analyze", {"url": url}, "a"),
        tool_call("scrape_api", {"url": url, "candidate_id": "api_0", "max_pages": 5}, "b"),
        # after the user says "yes":
        lambda m: tool_call("scrape_structure", {"dataset_id": last_dataset(m), "fields": ["name", "price"]}, "c"),
        lambda m: tool_call("scrape_save", {"dataset_id": last_dataset(m), "file_format": "csv",
                                            "filename": "products"}, "d"),
        AIMessage(content="Dynamic site, used its API, saved 24 rows to scrapes/products.csv"),
    ]
    first = run_chat(f"scrape product names and prices from {url}", "s_scrape")
    assert first["path"] == "playbook"
    assert first["status"] == "waiting_confirmation"
    second = run_chat("yes", "s_scrape")
    assert second["status"] == "success", second
    assert second["tools_used"] == ["scrape_api", "scrape_structure", "scrape_save"]
    csv = Path(get_settings().workspace_dir) / "scrapes" / "products.csv"
    lines = csv.read_text().strip().splitlines()
    assert lines[0] == "name,price" and len(lines) == 25
    assert "scrapes/products.csv" in second["files"]
