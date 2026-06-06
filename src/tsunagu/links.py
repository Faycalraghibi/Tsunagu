"""Page ids and internal link normalisation."""

from __future__ import annotations

import re
from dataclasses import dataclass

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
LANG_SEGMENTS = {"python", "javascript"}
EXT_RE = re.compile(r"\.(mdx|md)$")


@dataclass
class LinkStats:
    internal: int = 0
    external: int = 0
    out_of_corpus: int = 0


def page_id_from_repo_path(repo_path: str) -> str:
    """Map `src/oss/langgraph/persistence.mdx` to `langgraph/persistence`."""
    page = EXT_RE.sub("", repo_path.removeprefix("src/oss/"))
    return page.removesuffix("/index")


def normalize_link(href: str, source_page: str, known_pages: set[str]) -> tuple[str, str | None]:
    """Classify one link target.

    Returns:
        (kind, target) where kind is "internal", "external", "out_of_corpus" or "anchor",
        and target is a known page id for internal links, else None.
    """
    if href.startswith(("http://", "https://", "mailto:")):
        return "external", None
    if href.startswith("#"):
        return "anchor", None
    path = href.split("#", 1)[0].split("?", 1)[0]
    if path.startswith("/"):
        parts = [p for p in path.split("/") if p]
        if not parts or parts[0] != "oss":
            return "out_of_corpus", None
        parts = parts[1:]
    else:
        resolved: list[str] = source_page.split("/")[:-1]
        for part in path.split("/"):
            if part == "..":
                if resolved:
                    resolved.pop()
            elif part and part != ".":
                resolved.append(part)
        parts = resolved
    parts = [p for p in parts if p not in LANG_SEGMENTS]
    target = EXT_RE.sub("", "/".join(parts)).removesuffix("/index")
    if target in known_pages:
        return "internal", target
    return "out_of_corpus", None


def extract_links(
    text: str, source_page: str, known_pages: set[str]
) -> tuple[list[str], LinkStats]:
    """Find Markdown links in `text` (images excluded).

    Self-links are not returned as targets but are counted as internal.

    Returns:
        Unique internal target page ids in order of first appearance, and link counts.
    """
    targets: list[str] = []
    stats = LinkStats()
    for href in LINK_RE.findall(text):
        kind, target = normalize_link(href, source_page, known_pages)
        if kind == "internal" and target is not None:
            stats.internal += 1
            if target != source_page:
                targets.append(target)
        elif kind == "external":
            stats.external += 1
        elif kind == "out_of_corpus":
            stats.out_of_corpus += 1
    return list(dict.fromkeys(targets)), stats
