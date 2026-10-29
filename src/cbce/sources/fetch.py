"""HTTP fetching with a descriptive user agent and rate limiting. Standard library only."""

import os
import time
import urllib.request
from collections.abc import Callable

from cbce.sources.base import Fetcher, FetchResult

CONTACT_ENV = "CBCE_CONTACT"
TIMEOUT_SECONDS = 30


def user_agent() -> str:
    """Descriptive user agent. The contact comes from the environment, never from code."""
    contact = os.environ.get(CONTACT_ENV)
    if not contact:
        raise RuntimeError(f"set {CONTACT_ENV} to a contact address before fetching live pages")
    return f"cbce-research/0.1 (personal research project; contact: {contact})"


def http_fetch(url: str) -> FetchResult:
    """GET a URL, following redirects. Non-2xx responses raise."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return FetchResult(response.read(), response.geturl())


def rate_limited(
    fetch: Fetcher,
    min_interval: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Fetcher:
    """Wrap a fetcher so consecutive calls start at least `min_interval` seconds apart."""
    last_start: list[float] = []

    def limited(url: str) -> FetchResult:
        if last_start:
            wait = min_interval - (clock() - last_start[0])
            if wait > 0:
                sleep(wait)
        last_start[:] = [clock()]
        return fetch(url)

    return limited
