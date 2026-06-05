"""Turn Mintlify MDX into plain Markdown (Python variant only)."""

from __future__ import annotations

import re
from dataclasses import dataclass

FENCE_RE = re.compile(r"^\s*(```|~~~)")
FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.S)
API_REF_RE = re.compile(r"@\[`?([^`\]]+?)`?\](?:\[[^\]]*\])?")
JSX_LINE_RE = re.compile(r"^\s*</?(?:[A-Z][A-Za-z0-9]*|a)\b[^>]*>\s*$")
TITLE_ATTR_RE = re.compile(r'title="([^"]+)"')
IMPORT_RE = re.compile(r"^import\s+.+\s+from\s+['\"].+['\"];?$")
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.M)


@dataclass
class CleanPage:
    title: str
    text: str
    api_refs: list[str]


def _front_matter_title(block: str) -> str:
    for line in block.splitlines():
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""


def clean_mdx(raw: str) -> CleanPage:
    """Clean one MDX page.

    Drops front matter (keeping the title), comments, JSX tag lines, snippet imports and
    `:::js` blocks; unwraps `:::python` blocks; turns `@[`X`]` API references into code
    spans. Code fences are copied unchanged.

    Args:
        raw: The MDX source.

    Returns:
        The page title, the cleaned Markdown, and the sorted API reference names.
    """
    raw = raw.replace("\r\n", "\n")
    title = ""
    match = FRONT_MATTER_RE.match(raw)
    if match:
        title = _front_matter_title(match.group(1))
        raw = raw[match.end() :]
    raw = COMMENT_RE.sub("", raw)

    out: list[str] = []
    in_fence = False
    lang_block: str | None = None
    for line in raw.split("\n"):
        stripped = line.strip()
        if FENCE_RE.match(line):
            in_fence = not in_fence
            if lang_block != "js":
                out.append(line)
            continue
        if in_fence:
            if lang_block != "js":
                out.append(line)
            continue
        if stripped in (":::python", ":::js"):
            lang_block = stripped[3:]
            continue
        if stripped == ":::":  # closes a language block; a stray closer is dropped too
            lang_block = None
            continue
        if lang_block == "js" or IMPORT_RE.match(stripped):
            continue
        if JSX_LINE_RE.match(line):
            attr = TITLE_ATTR_RE.search(line)
            if attr and not stripped.startswith("</"):
                out.append(f"**{attr.group(1)}**")
            continue
        out.append(line)

    text = "\n".join(out)
    api_refs = sorted({m.strip() for m in API_REF_RE.findall(text)}, key=str.lower)
    text = API_REF_RE.sub(lambda m: f"`{m.group(1).strip()}`", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    if not title:
        h1 = H1_RE.search(text)
        title = h1.group(1) if h1 else ""
    return CleanPage(title=title, text=text, api_refs=api_refs)
