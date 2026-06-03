from __future__ import annotations

from pathlib import Path

import httpx

from tsunagu.corpus import RAW_URL, fetch_pages


def test_fetch_pages_writes_files_and_skips_existing(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text="# Title\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    pages = ["src/oss/langgraph/a.mdx", "src/oss/langgraph/b.mdx"]
    written = fetch_pages("org/repo", "abc123", pages, tmp_path, client=client)
    assert [p.name for p in written] == ["a.mdx", "b.mdx"]
    assert (tmp_path / "src/oss/langgraph/a.mdx").read_text(encoding="utf-8") == "# Title\n"
    assert seen[0] == RAW_URL.format(repo="org/repo", commit="abc123", path=pages[0])

    fetch_pages("org/repo", "abc123", pages, tmp_path, client=client)
    assert len(seen) == 2  # second call hit nothing: files already exist


def test_fetch_pages_force_refetches(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text=f"# Call {len(seen)}\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    pages = ["src/oss/langgraph/a.mdx"]
    fetch_pages("org/repo", "abc123", pages, tmp_path, client=client)
    assert len(seen) == 1
    assert (tmp_path / "src/oss/langgraph/a.mdx").read_text(encoding="utf-8") == "# Call 1\n"

    fetch_pages("org/repo", "abc123", pages, tmp_path, client=client, force=True)
    assert len(seen) == 2
    assert (tmp_path / "src/oss/langgraph/a.mdx").read_text(encoding="utf-8") == "# Call 2\n"
