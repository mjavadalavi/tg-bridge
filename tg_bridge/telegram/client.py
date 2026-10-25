"""TelegramClient Protocol + HttpTelegramClient (token per call, retries) (C04).

Port of the live file's tg() (C04). Every call re-reads the bot token from Paths.token, so a rotated token takes effect
without a restart. sendMessage, and editMessageText of the feed, first take a slot from the shared RateGate. HTTP 429
closes the gate for everyone for the server's retry_after and retries (FLOOD_RETRIES times). HTTP 400 is a rejected
request that never heals on a retry, so it raises TelegramBadRequest at once and the caller decides. Any other
failure is tried three times in all, five seconds apart, then raises TelegramError.

Exception texts never contain the request URL, so the bot token cannot reach the log through an error.
"""

import json
import urllib.error
import urllib.parse
from typing import Any, Protocol, runtime_checkable

from tg_bridge.clock import Clock
from tg_bridge.logging_ import Logger
from tg_bridge.paths import Paths
from tg_bridge.telegram.proxy import OpenerCache
from tg_bridge.telegram.ratelimit import RateGate

API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT = 65  # seconds; long enough for getUpdates long polling
FLOOD_RETRIES = 5  # HTTP 429 waits before a call gives up and counts as a failure
ATTEMPTS = 3  # tries for a failure that is neither 400 nor 429
RETRY_PAUSE = 5.0  # seconds between two tries of a failing call
BAD_REQUEST = 400
TOO_MANY_REQUESTS = 429
DESCRIPTION_LIMIT = 200
SEND_METHODS = ("sendMessage",)


class TelegramError(RuntimeError):
    """A Bot API call failed for good. code is the HTTP status when there was one, else None."""

    def __init__(
        self, method: str, code: int | None = None, description: str = "", retry_after: int | None = None
    ) -> None:
        self.method = method
        self.code = code
        self.description = description
        self.retry_after = retry_after
        super().__init__(f"{method} failed (code {code}): {description}".rstrip())


class TelegramBadRequest(TelegramError):
    """HTTP 400: the request was rejected (for example a message that does not parse). Never retried."""


@runtime_checkable
class TelegramClient(Protocol):
    """What the rest of the bridge may ask of Telegram. Implemented by HttpTelegramClient and FakeTelegram."""

    def call(self, method: str, feed: bool = False, **params: Any) -> Any:
        """One Bot API call. feed=True marks the live feed's send or edit, which takes only its share of the budget.

        Returns the 'result' field of the response (a dict for most methods, True for some).
        """
        ...


def _error_body(raw: bytes | None) -> dict[str, Any]:
    """The JSON object of a Telegram error body, or an empty dict when it is not one."""
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _description(body: dict[str, Any]) -> str:
    return str(body.get("description") or "")[:DESCRIPTION_LIMIT]


def _retry_after(body: dict[str, Any]) -> int | None:
    """The retry_after value of a 429 body, or None when it is missing or not a number."""
    parameters = body.get("parameters")
    value = parameters.get("retry_after") if isinstance(parameters, dict) else None
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


class HttpTelegramClient:
    """The real client: urllib over the shared egress proxy, with the send budget and flood-wait gate.

    Collaborators are injected: Paths (token file), Clock (retry pauses), Logger (retry lines), RateGate (the shared
    budget) and OpenerCache (proxy). api_base exists so tests can point the client at a local stub server.
    """

    def __init__(
        self,
        paths: Paths,
        clock: Clock,
        logger: Logger,
        rate: RateGate,
        openers: OpenerCache,
        api_base: str = API_BASE,
    ) -> None:
        self._paths = paths
        self._clock = clock
        self._logger = logger
        self._rate = rate
        self._openers = openers
        self._api_base = api_base.rstrip("/")

    def call(self, method: str, feed: bool = False, **params: Any) -> Any:
        token = self._read_token()
        url = f"{self._api_base}/bot{token}/{method}"
        data = urllib.parse.urlencode(params).encode() if params else None
        metered = method in SEND_METHODS or (method == "editMessageText" and feed)
        last: TelegramError | None = None
        failures = 0
        flood_waits = 0
        while True:
            if metered:
                self._rate.slot(feed)  # a feed edit takes its share like a send
            try:
                with self._openers.get().open(url, data=data, timeout=REQUEST_TIMEOUT) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict) or not payload.get("ok"):
                    raise TelegramError(method, None, str(payload)[:DESCRIPTION_LIMIT])
                return payload.get("result")
            except urllib.error.HTTPError as exc:
                body = _error_body(exc.read())
                if exc.code == BAD_REQUEST:
                    raise TelegramBadRequest(method, BAD_REQUEST, _description(body)) from None
                if exc.code == TOO_MANY_REQUESTS and flood_waits < FLOOD_RETRIES:
                    flood_waits += 1
                    seconds = self._rate.flood_wait(_retry_after(body))
                    self._logger.log(f"tg {method}: flood wait {seconds}s (429), retry {flood_waits}")
                    self._clock.sleep(seconds)
                    continue
                last = TelegramError(method, exc.code, _description(body), _retry_after(body))
            except Exception as exc:  # network errors and ok:false answers
                last = exc if isinstance(exc, TelegramError) else TelegramError(method, None, str(exc)[:DESCRIPTION_LIMIT])
            failures += 1
            if failures >= ATTEMPTS:
                assert last is not None
                raise last
            self._logger.log(f"tg {method}: attempt {failures} failed, retrying in {RETRY_PAUSE:g}s")
            self._clock.sleep(RETRY_PAUSE)

    def _read_token(self) -> str:
        with open(self._paths.token, encoding="utf-8") as f:
            return f.read().strip()
