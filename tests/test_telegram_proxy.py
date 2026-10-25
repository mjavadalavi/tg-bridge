"""Egress proxy resolution and the opener cache: port of the live proxy_url() and opener() (C04).

Order of sources: TG_EGRESS_PROXY_URL, then config.json 'egress_proxy', then the inbound named by
TG_EGRESS_PROXY_CONFIG and TG_EGRESS_PROXY_PORT, then a direct connection with one log line. The environment is
passed in, so no test reads or changes the real process environment.
"""

import json
import os
import unittest
import urllib.request

from tests.support import TempHome
from tg_bridge.paths import Paths
from tg_bridge.telegram.proxy import OpenerCache, proxy_url

INBOUND_USER = "bridge-user"
INBOUND_PASS = "bridge-secret"
INBOUND_PORT = 10820


def _write_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _inbound_config(port: int) -> dict[str, object]:
    return {
        "inbounds": [
            {"port": 1080, "settings": {"accounts": [{"user": "other", "pass": "x"}]}},
            {
                "port": port,
                "settings": {"accounts": [{"user": INBOUND_USER, "pass": INBOUND_PASS}]},
            },
        ]
    }


class ProxyUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = self.enterContext(TempHome())
        self.paths = Paths.from_home(self.home.home)
        self.inbound_path = os.path.join(self.home.home, "proxy.json")
        self.lines: list[str] = []

    def _resolve(self, env: dict[str, str]) -> str | None:
        return proxy_url(self.paths, env, self.lines.append)

    def test_explicit_env_url_wins_over_everything(self) -> None:
        _write_json(self.paths.config, {"egress_proxy": "http://from-config:1"})
        url = self._resolve({"TG_EGRESS_PROXY_URL": "http://from-env:2"})
        self.assertEqual(url, "http://from-env:2")

    def test_config_egress_proxy_is_next(self) -> None:
        _write_json(self.paths.config, {"egress_proxy": "http://from-config:3"})
        self.assertEqual(self._resolve({}), "http://from-config:3")

    def test_inbound_builds_the_credentialed_local_url(self) -> None:
        _write_json(self.inbound_path, _inbound_config(INBOUND_PORT))
        url = self._resolve(
            {"TG_EGRESS_PROXY_CONFIG": self.inbound_path, "TG_EGRESS_PROXY_PORT": str(INBOUND_PORT)}
        )
        self.assertEqual(url, f"http://{INBOUND_USER}:{INBOUND_PASS}@127.0.0.1:{INBOUND_PORT}")
        self.assertEqual(self.lines, [])

    def test_missing_port_or_no_matching_inbound_falls_back_direct_with_one_line(self) -> None:
        _write_json(self.inbound_path, _inbound_config(INBOUND_PORT))
        url = self._resolve({"TG_EGRESS_PROXY_CONFIG": self.inbound_path, "TG_EGRESS_PROXY_PORT": "9999"})
        self.assertIsNone(url)
        self.assertEqual(len(self.lines), 1)
        self.assertTrue(self.lines[0].startswith("proxy: fallback direct"))

    def test_nothing_configured_is_direct_without_a_crash(self) -> None:
        self.assertIsNone(self._resolve({}))
        self.assertEqual(len(self.lines), 1)

    def test_unreadable_config_is_skipped_quietly(self) -> None:
        with open(self.paths.config, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertIsNone(self._resolve({}))

    def test_fallback_line_never_carries_the_inbound_password(self) -> None:
        _write_json(self.inbound_path, {"inbounds": [{"port": 1, "settings": {"accounts": [{"user": "u", "pass": INBOUND_PASS}]}}]})
        self._resolve({"TG_EGRESS_PROXY_CONFIG": self.inbound_path, "TG_EGRESS_PROXY_PORT": "2"})
        self.assertNotIn(INBOUND_PASS, " ".join(self.lines))


class OpenerCacheTests(unittest.TestCase):
    def test_resolves_once_and_returns_the_same_opener(self) -> None:
        calls: list[int] = []

        def resolve() -> str | None:
            calls.append(1)
            return None

        cache = OpenerCache(resolve)
        first = cache.get()
        second = cache.get()
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)

    def test_proxy_is_installed_on_the_opener(self) -> None:
        cache = OpenerCache(lambda: "http://127.0.0.1:3128")
        handlers = [h for h in cache.get().handlers if isinstance(h, urllib.request.ProxyHandler)]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0].proxies.get("https"), "http://127.0.0.1:3128")

    def test_caches_are_independent_objects(self) -> None:
        self.assertIsNot(OpenerCache(lambda: None).get(), OpenerCache(lambda: None).get())


if __name__ == "__main__":
    unittest.main()
