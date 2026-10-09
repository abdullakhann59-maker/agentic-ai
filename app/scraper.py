"""Smart web scraper logic (Selenium + requests).

Flow:  analyze_site()  ->  recommended strategy  ->  one of
       extract_api() | extract_embedded() | extract_iframes() | extract_page()
       ->  structure_records()  ->  save_records()

The agent calls these through the tools in app/tools/scraper_tools.py, so the
AGENT decides which extraction to run based on the analysis report.
"""
from __future__ import annotations

import difflib
import json
import re
import time
import urllib.robotparser
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.browser import browser, domain_allowed
from app.config import get_settings

AD_DOMAINS = ("doubleclick", "googlesyndication", "adservice", "taboola", "outbrain", "facebook.com/tr")
DROP_HEADERS = {"cookie", "authorization", "content-length", "host", "connection"}


# =============================================================== analysis
def check_robots(url: str) -> tuple[bool, str]:
    s = get_settings()
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    try:
        resp = requests.get(robots_url, timeout=5, headers={"User-Agent": s.user_agent})
    except requests.RequestException:
        return True, "robots.txt not reachable (treated as allowed)"
    if resp.status_code >= 400:
        return True, "no robots.txt (allowed)"
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(resp.text.splitlines())
    ok = rp.can_fetch(s.user_agent, url)
    return ok, "allowed by robots.txt" if ok else "disallowed by robots.txt"


def analyze_site(url: str) -> dict:
    """Build a site profile: static/dynamic, APIs, embedded JSON, iframes, blockers."""
    profile: dict[str, Any] = {"url": url}

    # a) pre-checks ---------------------------------------------------------
    if not domain_allowed(url):
        profile.update(allowed=False, reason="Domain is not in ALLOWED_DOMAINS. Add it to .env to allow it.")
        return profile
    robots_ok, robots_msg = check_robots(url)
    profile["robots"] = robots_msg
    if not robots_ok:
        profile.update(allowed=False, reason=f"Scraping this path is {robots_msg}.")
        return profile
    profile["allowed"] = True

    # b) raw HTML vs rendered page -----------------------------------------
    s = get_settings()
    raw = requests.get(url, timeout=15, headers={"User-Agent": s.user_agent})
    raw_soup = BeautifulSoup(raw.text, "html.parser")
    for tag in raw_soup(["script", "style", "noscript"]):
        tag.decompose()
    raw_text = raw_soup.get_text(" ", strip=True)

    with browser.lock:
        browser.open(url)
        d = browser.driver()
        d.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        browser.wait_idle(idle_seconds=0.8, max_wait=6)
        rendered_text = d.find_element("tag name", "body").text
        page_source = d.page_source
        signals = _signals(d, page_source)
        iframes = _walk_iframes(d)
        network = list(browser.network_events)
        visible_values = _visible_values(rendered_text)
        api_candidates = _api_candidates(network, visible_values)

    raw_words = set(re.findall(r"\w+", raw_text.lower()))
    new_words = [w for w in dict.fromkeys(re.findall(r"\w+", rendered_text.lower())) if w not in raw_words]
    dynamic = len(new_words) >= 10 or len(rendered_text) > 1.5 * len(raw_text) + 200
    profile["site_type"] = "dynamic" if dynamic else "static"
    profile["evidence"] = (
        f"Rendered page has {len(new_words)} words that are not in the raw HTML"
        + (f" (e.g. {', '.join(new_words[:6])})" if new_words else "")
        + f". Raw text {len(raw_text)} chars vs rendered {len(rendered_text)} chars."
    )

    # c) APIs ----------------------------------------------------------------
    profile["api_candidates"] = api_candidates
    if dynamic and not api_candidates:
        profile["api_note"] = "No usable API found; site is highly dynamic. Use rendered-page extraction."

    # d) embedded JSON -------------------------------------------------------
    profile["embedded_data"] = _embedded_json(page_source)

    # e) iframes ---------------------------------------------------------------
    profile["iframes"] = iframes
    profile["iframe_count"] = len(iframes)

    # f) signals -------------------------------------------------------------
    profile["signals"] = signals

    # g) recommendation -------------------------------------------------------
    profile["recommended"] = _recommend(profile)
    return profile


def _visible_values(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    vals = [ln for ln in dict.fromkeys(lines) if 3 <= len(ln) <= 60]
    return vals[:300]


def _api_candidates(events: list[dict], visible_values: list[str]) -> list[dict]:
    requests_by_id: dict[str, dict] = {}
    responses: dict[str, dict] = {}
    for ev in events:
        p = ev.get("params", {})
        rid = p.get("requestId")
        if ev["method"] == "Network.requestWillBeSent":
            requests_by_id[rid] = p.get("request", {})
        elif ev["method"] == "Network.responseReceived":
            responses[rid] = p
    candidates = []
    for rid, resp in responses.items():
        r = resp.get("response", {})
        if resp.get("type") not in ("XHR", "Fetch") or "json" not in r.get("mimeType", ""):
            continue
        body = browser.response_body(rid)
        if not body:
            continue
        try:
            data = json.loads(body)
        except ValueError:
            continue
        records, path = find_records(data)
        matched = [v for v in visible_values if v in body]
        req = requests_by_id.get(rid, {})
        candidates.append({
            "url": r.get("url"),
            "method": req.get("method", "GET"),
            "status": r.get("status"),
            "headers": {k: v for k, v in (req.get("headers") or {}).items() if k.lower() not in DROP_HEADERS},
            "score": len(matched) + (5 if records else 0),
            "matched_values": len(matched),
            "records_found": len(records),
            "records_path": path,
            "sample": records[:2] if records else str(data)[:200],
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    for i, c in enumerate(candidates[:5]):
        c["id"] = f"api_{i}"
    return candidates[:5]


def find_records(data: Any, path: str = "$") -> tuple[list[dict], str]:
    """Find the largest list of dicts inside a JSON value (that is usually the data rows)."""
    best: tuple[list[dict], str] = ([], "")
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        best = (data, path)
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = ((str(i), v) for i, v in enumerate(data[:50]))
    else:
        items = ()
    for k, v in items:
        if isinstance(v, (dict, list)):
            cand = find_records(v, f"{path}.{k}")
            if len(cand[0]) > len(best[0]):
                best = cand
    return best


def _embedded_json(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    found = []
    blobs: list[tuple[str, str]] = []
    for sc in soup.find_all("script"):
        txt = sc.string or sc.get_text() or ""
        if sc.get("type") == "application/ld+json":
            blobs.append(("json-ld", txt))
        elif sc.get("id") == "__NEXT_DATA__":
            blobs.append(("__NEXT_DATA__", txt))
        else:
            m = re.search(r"window\.(__[A-Z_]+__)\s*=\s*(\{.*?\})\s*;?\s*$", txt, re.S)
            if m:
                blobs.append((m.group(1), m.group(2)))
    for i, (kind, txt) in enumerate(blobs):
        try:
            data = json.loads(txt)
        except ValueError:
            continue
        records, path = find_records(data)
        found.append({"id": f"emb_{i}", "kind": kind, "records_found": len(records), "records_path": path,
                      "records": records, "sample": records[:2]})
    return found


def _walk_iframes(d, depth: int = 0, prefix: str = "") -> list[dict]:
    """Visit every iframe (recursively), record its details, always return to the parent."""
    out: list[dict] = []
    if depth > 3:
        return out
    frames = d.find_elements("tag name", "iframe")
    for i, f in enumerate(frames):
        fid = f"{prefix}{i}"
        src = f.get_attribute("src") or "(inline)"
        size = f.size
        area = size.get("width", 0) * size.get("height", 0)
        visible = f.is_displayed()
        kind = "ad/tracker" if (area < 2500 or not visible or any(a in src for a in AD_DOMAINS)) else "content"
        info = {"id": fid, "src": src, "width": size.get("width"), "height": size.get("height"),
                "kind": kind, "depth": depth}
        try:
            d.switch_to.frame(f)
            info["text_chars"] = len(d.find_element("tag name", "body").text)
            out.append(info)
            out.extend(_walk_iframes(d, depth + 1, prefix=f"{fid}."))
        except Exception as e:  # cross-origin or detached frames
            info["error"] = str(e)[:100]
            out.append(info)
        finally:
            d.switch_to.parent_frame()
    return out


def _signals(d, html: str) -> dict:
    low = html.lower()
    links = d.execute_script(
        "return Array.from(document.querySelectorAll('a,button')).map(e => (e.innerText||'').trim().toLowerCase())")
    return {
        "captcha": any(k in low for k in ("captcha", "recaptcha", "hcaptcha", "verify you are human")),
        "login_wall": bool(d.find_elements("css selector", "input[type=password]")),
        "load_more": any(("load more" in t or "show more" in t) for t in links),
        "pagination": any(t in ("next", "›", "»", "next page") for t in links) or "page=" in low,
    }


def _recommend(p: dict) -> dict:
    sig = p["signals"]
    if sig["captcha"] or sig["login_wall"]:
        what = "a CAPTCHA" if sig["captcha"] else "a login wall"
        return {"strategy": "stop", "reason": f"The page shows {what}. Ask the user; never bypass it."}
    good_api = [c for c in p["api_candidates"] if c["records_found"] and c["matched_values"] >= 3]
    if good_api:
        c = good_api[0]
        return {"strategy": "api", "candidate_id": c["id"],
                "reason": f"API {c['url']} returns {c['records_found']} records and contains "
                          f"{c['matched_values']} values visible on the page. Cleanest and fastest source."}
    emb = [e for e in p["embedded_data"] if e["records_found"]]
    if emb:
        return {"strategy": "embedded", "embed_id": emb[0]["id"],
                "reason": f"The page embeds {emb[0]['records_found']} records as JSON ({emb[0]['kind']})."}
    content_frames = [f for f in p["iframes"] if f["kind"] == "content"]
    if content_frames:
        return {"strategy": "iframes", "iframe_ids": [f["id"] for f in content_frames],
                "reason": f"{len(content_frames)} content iframes hold the data "
                          f"({p['iframe_count'] - len(content_frames)} ad/tracker frames skipped)."}
    return {"strategy": "page",
            "reason": "Extract directly from the " + ("rendered page (JavaScript-built)." if p["site_type"] == "dynamic"
                                                     else "page HTML (simple static site).")}


def profile_summary(p: dict) -> dict:
    """Short version of the profile for the LLM (no big samples)."""
    if not p.get("allowed"):
        return {"url": p["url"], "allowed": False, "reason": p.get("reason")}
    return {
        "url": p["url"],
        "site_type": p["site_type"],
        "evidence": p["evidence"],
        "robots": p["robots"],
        "api_candidates": [{k: c[k] for k in ("id", "url", "method", "score", "records_found", "matched_values")}
                           for c in p["api_candidates"]],
        "api_note": p.get("api_note"),
        "embedded_data": [{k: e[k] for k in ("id", "kind", "records_found")} for e in p["embedded_data"]],
        "iframe_count": p["iframe_count"],
        "iframes": [{k: f.get(k) for k in ("id", "src", "kind", "depth")} for f in p["iframes"]],
        "signals": p["signals"],
        "recommended": p["recommended"],
    }


# ============================================================= extraction
def extract_api(profile: dict, candidate_id: str, max_pages: int) -> list[dict]:
    s = get_settings()
    cand = next((c for c in profile.get("api_candidates", []) if c["id"] == candidate_id), None)
    if cand is None:
        raise ValueError(f"Unknown API candidate {candidate_id}")
    if cand["method"].upper() != "GET":
        raise ValueError("Only GET APIs are replayed (POST could change data on the site).")
    parts = urlparse(cand["url"])
    query = {k: v[0] for k, v in parse_qs(parts.query).items()}
    headers = {k: v for k, v in cand["headers"].items() if not k.startswith(":")}
    headers["User-Agent"] = s.user_agent
    page_key = next((k for k in ("page", "p", "pageNumber") if k in query), None)
    start = int(query.get(page_key, 1)) if page_key else 1
    records: list[dict] = []
    for i in range(max_pages):
        if page_key:
            query[page_key] = str(start + i)
        url = urlunparse(parts._replace(query=urlencode(query)))
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200 or "json" not in resp.headers.get("content-type", ""):
            break
        data = resp.json()
        page_records, _ = find_records(data)
        if not page_records:
            break
        records.extend(page_records)
        total_pages = data.get("total_pages") if isinstance(data, dict) else None
        if not page_key or (total_pages and start + i >= int(total_pages)):
            break
        time.sleep(s.scrape_delay_seconds)   # polite rate limit
    return records


def extract_embedded(profile: dict, embed_id: str) -> list[dict]:
    emb = next((e for e in profile.get("embedded_data", []) if e["id"] == embed_id), None)
    if emb is None:
        raise ValueError(f"Unknown embedded data id {embed_id}")
    return list(emb["records"])


def extract_iframes(url: str, iframe_ids: list[str]) -> list[dict]:
    records: list[dict] = []
    with browser.lock:
        browser.open(url)
        d = browser.driver()
        for fid in iframe_ids:
            d.switch_to.default_content()
            try:
                for idx in fid.split("."):
                    d.switch_to.frame(d.find_elements("tag name", "iframe")[int(idx)])
            except (IndexError, ValueError):
                continue
            src = d.execute_script("return location.href")
            records.extend({"_frame": fid, "_source": src} | r for r in _records_from_html(d.page_source))
        d.switch_to.default_content()
    return records


def extract_page(url: str, max_pages: int) -> list[dict]:
    with browser.lock:
        browser.open(url)
        d = browser.driver()
        for _ in range(max(0, max_pages - 1)):          # click "load more" if present
            btns = [b for b in d.find_elements("css selector", "button, a")
                    if b.is_displayed() and ("load more" in b.text.lower() or "show more" in b.text.lower())]
            if not btns:
                break
            btns[0].click()
            browser.wait_idle(idle_seconds=0.8, max_wait=6)
        html = d.page_source
    records = _records_from_html(html)
    if not records:
        records = _cards_from_html(html)
    if not records:
        soup = BeautifulSoup(html, "html.parser")
        records = [{"text": p.get_text(" ", strip=True)} for p in soup.find_all(["p", "h1", "h2", "h3", "li"])
                   if p.get_text(strip=True)]
    return records


def _records_from_html(html: str) -> list[dict]:
    """Tables first (pandas.read_html), then list items."""
    records: list[dict] = []
    if "<table" in html.lower():
        try:
            for t_idx, df in enumerate(pd.read_html(StringIO(html))):
                df.columns = [str(c) for c in df.columns]
                for row in df.to_dict(orient="records"):
                    records.append({"_table": t_idx} | {k: _plain(v) for k, v in row.items()})
        except ValueError:
            pass
    soup = BeautifulSoup(html, "html.parser")
    for li in soup.find_all("li"):
        txt = li.get_text(" ", strip=True)
        if txt:
            records.append({"item": txt})
    return records


def _cards_from_html(html: str) -> list[dict]:
    """Find repeated 'cards' (same CSS class used 3+ times) and read their child fields."""
    soup = BeautifulSoup(html, "html.parser")
    counts: dict[str, int] = {}
    for el in soup.find_all(class_=True):
        for c in el.get("class", []):
            counts[c] = counts.get(c, 0) + 1
    best = None
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        if n < 3:
            break
        els = soup.find_all(class_=cls)
        if all(len(e.find_all(True)) >= 1 for e in els[:5]):
            best = els
            break
    if not best:
        return []
    rows = []
    for el in best:
        row = {}
        for child in el.find_all(class_=True):
            row[child["class"][0]] = child.get_text(" ", strip=True)
        rows.append(row or {"text": el.get_text(" ", strip=True)})
    return rows


def _plain(v: Any) -> Any:
    if isinstance(v, float) and pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


# ============================================================ structuring
SYNONYMS = {
    "name": ["title", "product", "product_name", "item", "name"],
    "price": ["price", "price_inr", "cost", "amount", "mrp"],
    "rating": ["rating", "stars", "score"],
    "category": ["category", "type", "department"],
}


def map_fields(keys: list[str], fields: list[str]) -> dict[str, str | None]:
    """Deterministic field mapping (synonyms + fuzzy match). Unmapped fields -> None."""
    mapping: dict[str, str | None] = {}
    low = {k.lower(): k for k in keys}
    for f in fields:
        fl = f.lower().strip()
        choice = None
        for cand in [fl] + SYNONYMS.get(fl, []):
            if cand in low:
                choice = low[cand]
                break
        if choice is None:
            close = difflib.get_close_matches(fl, list(low), n=1, cutoff=0.6)
            choice = low[close[0]] if close else None
        if choice is None:
            sub = [k for k in low if fl in k or k in fl]
            choice = low[sub[0]] if sub else None
        mapping[f] = choice
    return mapping


def clean_value(v: Any) -> Any:
    if isinstance(v, str):
        t = v.replace("₹", "").replace("Rs.", "").replace("$", "").replace("★", "").replace(",", "").strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", t):
            return float(t) if "." in t else int(t)
        return v.strip()
    return v


def structure_records(records: list[dict], fields: list[str], llm_mapper=None) -> tuple[list[dict], dict]:
    keys = list(dict.fromkeys(k for r in records[:50] for k in r if not k.startswith("_")))
    mapping = map_fields(keys, fields) if fields else {k: k for k in keys}
    missing = [f for f, k in mapping.items() if k is None]
    if missing and llm_mapper is not None:
        mapping.update({f: k for f, k in llm_mapper(keys, missing, records[:2]).items() if k in keys})
    out, seen = [], set()
    for r in records:
        row = {f: clean_value(r.get(k)) if k else None for f, k in mapping.items()}
        if all(v in (None, "") for v in row.values()):
            continue
        sig = json.dumps(row, sort_keys=True, default=str)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out, mapping


def save_records(records: list[dict], fmt: str, filename: str, meta: dict) -> Path:
    s = get_settings()
    folder = Path(s.workspace_dir) / "scrapes"
    folder.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem) or "scrape"
    fmt = fmt.lower()
    path = folder / f"{stem}.{fmt}"
    df = pd.DataFrame(records)
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    elif fmt == "xlsx":
        df.to_excel(path, index=False)
    else:
        raise ValueError("format must be csv, json or xlsx")
    (folder / f"{stem}.meta.json").write_text(json.dumps(meta | {"rows": len(records)}, indent=2, default=str))
    return path
