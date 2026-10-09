"""Smart scraper tools. Large data stays in the session store; the LLM gets ids + summaries."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from app import scraper as sc
from app.context import ctx, session_store
from app.llm import get_chat_model, token_usage
from app.tools.confirm import register, request_confirmation


def _profile(url: str) -> dict:
    p = session_store.get_profile(url)
    if p is None:
        p = sc.analyze_site(url)
        session_store.save_profile(url, p)
    return p


def _dataset_reply(records: list[dict], method: str, url: str) -> str:
    ds_id = session_store.save_dataset(records, {"url": url, "method": method})
    cols = list(dict.fromkeys(k for r in records[:20] for k in r))
    return json.dumps({"dataset_id": ds_id, "rows": len(records), "method": method, "columns": cols,
                       "sample": records[:3]}, default=str, ensure_ascii=False)


@tool
def scrape_analyze(url: str) -> str:
    """ALWAYS call first when asked to scrape/extract data from a website. Checks robots.txt and the
    allowlist, detects static vs dynamic (JavaScript) pages, finds hidden JSON APIs from the browser's
    network traffic, embedded JSON, iframes (count, nested, ads) and CAPTCHA/login walls, and returns
    the recommended strategy: api, embedded, iframes, page or stop."""
    p = sc.analyze_site(url)
    session_store.save_profile(url, p)
    return json.dumps(sc.profile_summary(p), ensure_ascii=False, default=str)


@tool
def scrape_api(url: str, candidate_id: str = "api_0", max_pages: int = 5) -> str:
    """Fetch data directly from a site's detected JSON API (GET only, rate limited, follows pagination).
    Needs user confirmation. Use only when scrape_analyze recommended strategy 'api'."""
    p = _profile(url)
    cand = next((c for c in p.get("api_candidates", []) if c["id"] == candidate_id), None)
    if cand is None:
        return f"Error: no API candidate {candidate_id} for this URL. Run scrape_analyze first."
    return request_confirmation(
        "scrape_api", {"url": url, "candidate_id": candidate_id, "max_pages": max_pages},
        f"Call the site's API {cand['url']} (GET, up to {max_pages} pages, 1 request/second).",
        continue_after=True)


@register("scrape_api")
def _run_scrape_api(url: str, candidate_id: str, max_pages: int) -> str:
    records = sc.extract_api(_profile(url), candidate_id, max_pages)
    return _dataset_reply(records, "api", url)


@tool
def scrape_embedded(url: str, embed_id: str = "emb_0") -> str:
    """Read data embedded as JSON inside the page (e.g. __NEXT_DATA__, JSON-LD).
    Use when scrape_analyze recommended strategy 'embedded'."""
    return _dataset_reply(sc.extract_embedded(_profile(url), embed_id), "embedded_json", url)


@tool
def scrape_iframes(url: str, iframe_ids: list[str] | None = None) -> str:
    """Switch into iframes one by one (including nested ones) and extract their tables and lists.
    Use when scrape_analyze recommended strategy 'iframes'. If iframe_ids is empty, all content
    iframes are used (ads/trackers skipped)."""
    p = _profile(url)
    ids = iframe_ids or [f["id"] for f in p.get("iframes", []) if f["kind"] == "content"]
    return _dataset_reply(sc.extract_iframes(url, ids), "iframes", url)


@tool
def scrape_page(url: str, max_pages: int = 3) -> str:
    """Extract tables, repeated product cards or text from the rendered page (works for static and
    JavaScript sites; clicks 'load more' up to max_pages). Use when strategy is 'page', or when no API
    is available."""
    return _dataset_reply(sc.extract_page(url, max_pages), "rendered_page", url)


def _llm_mapper(keys: list[str], fields: list[str], sample: list[dict]) -> dict:
    llm = get_chat_model(json_mode=True)
    prompt = (f"Map each wanted field to one of the available columns.\nWanted fields: {fields}\n"
              f"Available columns: {keys}\nSample rows: {json.dumps(sample, default=str)[:800]}\n"
              'Return JSON like {"field": "column or null"}.')
    msg = llm.invoke(prompt)
    ctx().add_tokens(*token_usage(msg))
    try:
        return json.loads(msg.content)
    except (ValueError, TypeError):
        return {}


@tool
def scrape_structure(dataset_id: str, fields: list[str] | None = None) -> str:
    """Clean a scraped dataset: keep/rename only the wanted fields (e.g. ["name","price","rating"]),
    convert prices and numbers, and remove duplicates. Returns a new dataset_id."""
    ds = session_store.get_dataset(dataset_id)
    if ds is None:
        return f"Error: unknown dataset {dataset_id}"
    rows, mapping = sc.structure_records(ds["records"], fields or [], llm_mapper=_llm_mapper)
    new_id = session_store.save_dataset(rows, ds["meta"] | {"mapping": mapping})
    return json.dumps({"dataset_id": new_id, "rows": len(rows), "mapping": mapping, "sample": rows[:3]},
                      default=str, ensure_ascii=False)


@tool
def scrape_save(dataset_id: str, file_format: str = "csv", filename: str = "scraped_data") -> str:
    """Save a dataset to a file (csv, json or xlsx) in the workspace and return the path and row count."""
    ds = session_store.get_dataset(dataset_id)
    if ds is None:
        return f"Error: unknown dataset {dataset_id}"
    from app.config import get_settings

    path = sc.save_records(ds["records"], file_format, filename, ds["meta"])
    rel = str(path.relative_to(get_settings().workspace_dir))
    ctx().add_file(rel)
    return f"Saved {len(ds['records'])} rows to {rel} (method: {ds['meta'].get('method')})."
