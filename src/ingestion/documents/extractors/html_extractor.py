"""HTML extraction: strip boilerplate and return main-content text via BeautifulSoup."""

from __future__ import annotations

from bs4 import BeautifulSoup

_BOILERPLATE_TAGS = ["script", "style", "nav", "footer", "header"]


def extract_html(content: bytes) -> str:
    """
    Parse HTML bytes, remove boilerplate tags, and return the visible text from
    the main content area (<main>, <article>, or <body> as fallback).
    """
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    return (main or soup).get_text(separator="\n", strip=True)
