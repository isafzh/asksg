"""Playwright transport layer: render JavaScript-heavy pages and return raw HTML bytes."""

from __future__ import annotations

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def fetch_with_playwright(url: str, wait_ms: int = 3000) -> bytes | None:
    """
    Navigate to `url` with a headless Chromium browser and return the fully
    rendered HTML as bytes.  Returns None if Playwright is not installed or
    navigation fails.

    Install: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=_USER_AGENT,
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        try:
            page.goto(url, timeout=60_000, wait_until="load")
            page.wait_for_timeout(wait_ms)
            return page.content().encode()
        except Exception as e:
            print(f"  Playwright error: {e}")
            return None
        finally:
            browser.close()
