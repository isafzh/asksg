"""HTTP transport layer: download any URL and return raw bytes."""

from __future__ import annotations

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_url(urls: list[str], timeout: int = 30) -> bytes | None:
    """Try each URL in order; return raw bytes on first 200, None if all fail."""
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                print(f"  OK: {url}")
                return r.content
            print(f"  HTTP {r.status_code}: {url}")
        except requests.RequestException as e:
            print(f"  Error: {e}")
    return None
