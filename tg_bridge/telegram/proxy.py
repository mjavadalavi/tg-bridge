"""Egress proxy URL and the urllib opener cache (C04).

proxy_url() ports the live file's proxy_url(): an explicit TG_EGRESS_PROXY_URL wins, then the 'egress_proxy' key of
config.json, then the local HTTP-proxy inbound named by TG_EGRESS_PROXY_CONFIG (matched on TG_EGRESS_PROXY_PORT).
Nothing is read at import time: the environment and the config path are passed in, so a test can point them at a
temp directory. OpenerCache builds the opener once per client and keeps it as an instance attribute (no module
global), resolving the proxy lazily on first use as the live opener() did.
"""

import json
import threading
import urllib.request
from collections.abc import Callable, Mapping

from tg_bridge.paths import Paths

LogFn = Callable[[str], None]
PROXY_ENV = "TG_EGRESS_PROXY_URL"
PROXY_CONFIG_ENV = "TG_EGRESS_PROXY_CONFIG"
PROXY_PORT_ENV = "TG_EGRESS_PROXY_PORT"
CONFIG_PROXY_KEY = "egress_proxy"


def _load_json(path: str) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _inbound_proxy(config_path: str, port: str) -> str:
    """The http://user:pass@127.0.0.1:<port> URL of the inbound listening on port. Raises when it is not there."""
    inbounds = _load_json(config_path)["inbounds"]  # type: ignore[index]
    inbound = next(i for i in inbounds if str(i.get("port")) == port)
    account = inbound["settings"]["accounts"][0]
    return f"http://{account['user']}:{account['pass']}@127.0.0.1:{port}"


def proxy_url(paths: Paths, env: Mapping[str, str], log: LogFn) -> str | None:
    """The egress proxy to use for Telegram and Soniox traffic, or None for a direct connection.

    Order: TG_EGRESS_PROXY_URL, config.json 'egress_proxy', then the inbound named by TG_EGRESS_PROXY_CONFIG and
    TG_EGRESS_PROXY_PORT. A failed lookup of the last one logs one line and falls back to a direct connection.
    """
    explicit = env.get(PROXY_ENV)
    if explicit:
        return explicit
    try:
        configured = (_load_json(paths.config) or {}).get(CONFIG_PROXY_KEY)  # type: ignore[union-attr]
        if configured:
            return str(configured)
    except Exception:
        pass
    port = env.get(PROXY_PORT_ENV, "0")
    try:
        return _inbound_proxy(env.get(PROXY_CONFIG_ENV, ""), port)
    except Exception as exc:
        log(f"proxy: fallback direct ({exc})")
        return None


class OpenerCache:
    """Builds one urllib opener on first use and returns it on every later call. Thread-safe."""

    def __init__(self, resolve_proxy: Callable[[], str | None]) -> None:
        self._resolve_proxy = resolve_proxy
        self._lock = threading.Lock()
        self._opener: urllib.request.OpenerDirector | None = None

    def get(self) -> urllib.request.OpenerDirector:
        with self._lock:
            if self._opener is None:
                self._opener = self._build(self._resolve_proxy())
            return self._opener

    @staticmethod
    def _build(proxy: str | None) -> urllib.request.OpenerDirector:
        if proxy:
            return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        return urllib.request.build_opener()
