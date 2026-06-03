"""Download the pinned documentation pages from GitHub."""

from __future__ import annotations

from pathlib import Path

import httpx

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def fetch_pages(
    repo: str,
    commit: str,
    pages: list[str],
    raw_dir: Path,
    *,
    client: httpx.Client | None = None,
    force: bool = False,
) -> list[Path]:
    """Download each page at the pinned commit into `raw_dir`, keeping repo paths.

    Args:
        repo: GitHub "owner/name".
        commit: Full commit hash.
        pages: Repo-relative file paths.
        raw_dir: Destination root.
        client: Optional httpx client (tests inject a mock transport).
        force: Re-download files that already exist.

    Returns:
        The local paths, in the order of `pages`.
    """
    own_client = client is None
    http = client or httpx.Client(timeout=30.0, follow_redirects=True)
    written: list[Path] = []
    try:
        for repo_path in pages:
            dest = raw_dir / repo_path
            if dest.exists() and not force:
                written.append(dest)
                continue
            response = http.get(RAW_URL.format(repo=repo, commit=commit, path=repo_path))
            response.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(response.text, encoding="utf-8")
            written.append(dest)
    finally:
        if own_client:
            http.close()
    return written
