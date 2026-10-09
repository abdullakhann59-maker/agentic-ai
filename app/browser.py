"""One shared Selenium Chrome driver for the browser and scraper tools.

Performance logging is switched on so the scraper can read the browser's network
traffic (that is how hidden APIs are detected).
"""
from __future__ import annotations

import json
import threading
import time
from urllib.parse import urlparse

from app.config import get_settings


class BrowserManager:
    def __init__(self) -> None:
        self._driver = None
        self.lock = threading.RLock()
        self.network_events: list[dict] = []
        self.elements: dict[int, object] = {}   # ids shown to the LLM by browser_find

    # ------------------------------------------------------------ driver
    def driver(self):
        with self.lock:
            if self._driver is None:
                self._driver = self._start()
            return self._driver

    def _start(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service

        s = get_settings()
        opts = webdriver.ChromeOptions()
        if s.chrome_binary:
            opts.binary_location = s.chrome_binary
        if s.headless:
            opts.add_argument("--headless=new")
        for arg in ("--no-sandbox", "--disable-dev-shm-usage", "--window-size=1366,900",
                    f"--user-agent={s.user_agent}"):
            opts.add_argument(arg)
        opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        service_args = s.chromedriver_args.split() if s.chromedriver_args else None
        service = Service(s.chromedriver_path or None, service_args=service_args)
        drv = webdriver.Chrome(options=opts, service=service)
        drv.set_page_load_timeout(30)
        drv.execute_cdp_cmd("Network.enable", {})
        return drv

    def close(self) -> None:
        with self.lock:
            if self._driver is not None:
                try:
                    self._driver.quit()
                finally:
                    self._driver = None

    # ----------------------------------------------------------- network
    def drain_network(self) -> list[dict]:
        """Read new performance-log entries and keep the Network.* events."""
        new = []
        for entry in self.driver().get_log("performance"):
            try:
                msg = json.loads(entry["message"])["message"]
            except (KeyError, ValueError):
                continue
            if msg.get("method", "").startswith("Network."):
                new.append(msg)
        self.network_events.extend(new)
        return new

    def open(self, url: str, idle_seconds: float = 1.0, max_wait: float = 10.0) -> None:
        """Navigate and wait until the page is loaded AND the network is quiet.

        This replaces fixed sleeps: we poll the network log and stop when no new
        requests happened for `idle_seconds` (or `max_wait` is reached).
        """
        d = self.driver()
        self.drain_network()
        self.network_events = []
        self.elements = {}
        d.get(url)
        self.wait_idle(idle_seconds, max_wait)

    def wait_idle(self, idle_seconds: float = 1.0, max_wait: float = 10.0) -> None:
        d = self.driver()
        start = last = time.time()
        while time.time() - start < max_wait:
            if self.drain_network():
                last = time.time()
            ready = d.execute_script("return document.readyState") == "complete"
            if ready and time.time() - last >= idle_seconds:
                return
            time.sleep(0.2)

    def response_body(self, request_id: str) -> str | None:
        try:
            res = self.driver().execute_cdp_cmd("Network.getResponseBody", {"requestId": request_id})
            return res.get("body")
        except Exception:
            return None


browser = BrowserManager()


def domain_allowed(url: str) -> bool:
    allowed = get_settings().allowed_domain_list
    if "*" in allowed:
        return True
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in allowed)


def selenium_ready() -> bool:
    try:
        import selenium  # noqa: F401
        return True
    except ImportError:
        return False
