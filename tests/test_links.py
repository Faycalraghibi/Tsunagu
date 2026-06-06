from __future__ import annotations

from tsunagu.links import extract_links, normalize_link, page_id_from_repo_path

KNOWN = {
    "langgraph/persistence",
    "langgraph/streaming",
    "langgraph/interrupts",
    "concepts/memory",
    "langchain/multi-agent",
}


def test_page_ids() -> None:
    assert page_id_from_repo_path("src/oss/langgraph/persistence.mdx") == "langgraph/persistence"
    assert (
        page_id_from_repo_path("src/oss/langchain/multi-agent/index.mdx") == "langchain/multi-agent"
    )
    assert (
        page_id_from_repo_path("src/oss/langgraph/frontend/overview.md")
        == "langgraph/frontend/overview"
    )


def test_normalize_link_kinds() -> None:
    src = "langgraph/interrupts"
    assert normalize_link("/oss/langgraph/persistence#threads", src, KNOWN) == (
        "internal",
        "langgraph/persistence",
    )
    assert normalize_link("/oss/python/langgraph/streaming", src, KNOWN) == (
        "internal",
        "langgraph/streaming",
    )
    assert normalize_link("/oss/langchain/multi-agent", src, KNOWN) == (
        "internal",
        "langchain/multi-agent",
    )
    assert normalize_link("../concepts/memory", src, KNOWN) == ("internal", "concepts/memory")
    assert normalize_link("https://smith.langchain.com", src, KNOWN) == ("external", None)
    assert normalize_link("/langsmith/agent-server", src, KNOWN) == ("out_of_corpus", None)
    assert normalize_link("/oss/langgraph/unknown-page", src, KNOWN) == ("out_of_corpus", None)
    assert normalize_link("#local-anchor", src, KNOWN) == ("anchor", None)


def test_normalize_link_edge_cases() -> None:
    src = "langgraph/interrupts"
    # Query parameters and anchors
    assert normalize_link("/oss/langgraph/persistence?version=1#threads", src, KNOWN) == (
        "internal",
        "langgraph/persistence",
    )
    # Mailto link is external
    assert normalize_link("mailto:support@langchain.com", src, KNOWN) == ("external", None)
    # Relative path with current dir segment ./
    assert normalize_link("./persistence", src, KNOWN) == ("internal", "langgraph/persistence")
    # Empty path or root slash
    assert normalize_link("/", src, KNOWN) == ("out_of_corpus", None)


def test_extract_links_dedupes_and_counts() -> None:
    text = (
        "See [persistence](/oss/langgraph/persistence) and [again](/oss/langgraph/persistence#x), "
        "[LangSmith](https://smith.langchain.com), [server](/langsmith/agent-server), "
        "[self](/oss/langgraph/interrupts) and ![img](/oss/langgraph/streaming)."
    )
    targets, stats = extract_links(text, "langgraph/interrupts", KNOWN)
    assert targets == ["langgraph/persistence"]
    assert (stats.internal, stats.external, stats.out_of_corpus) == (
        3,
        1,
        1,
    )  # self-link counted, not returned


def test_extract_links_preserves_order_and_handles_titles() -> None:
    text = (
        'Read [memory](../concepts/memory "Memory Concepts") then '
        "[streaming](/oss/python/langgraph/streaming) before "
        "[persistence](/oss/langgraph/persistence) and "
        "[memory again](../concepts/memory)."
    )
    targets, stats = extract_links(text, "langgraph/interrupts", KNOWN)
    assert targets == ["concepts/memory", "langgraph/streaming", "langgraph/persistence"]
    assert stats.internal == 4
