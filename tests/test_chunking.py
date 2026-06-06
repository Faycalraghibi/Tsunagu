from __future__ import annotations

from pathlib import Path

from tsunagu.chunking import (
    Chunk,
    chunk_page,
    extract_symbols,
    load_chunks,
    normalize_symbol,
    save_chunks,
)
from tsunagu.config import ChunkingConfig
from tsunagu.tokens import count_tokens

CFG = ChunkingConfig(target_tokens=120, overlap_tokens=20, max_tokens=200)


def _prose(n_words: int, word: str = "alpha") -> str:
    return " ".join([word] * n_words) + "."


def _fence_lines_balanced(text: str) -> bool:
    return sum(1 for line in text.splitlines() if line.strip().startswith("```")) % 2 == 0


def test_heading_path_includes_title_and_nesting() -> None:
    text = "# Persistence\n\nIntro.\n\n## Checkpointers\n\nA.\n\n### Memory saver\n\nB.\n"
    chunks = chunk_page("langgraph/persistence", "Persistence", text, CFG, set())
    paths = [c.heading_path for c in chunks]
    assert ["Persistence"] in paths
    assert ["Persistence", "Checkpointers", "Memory saver"] in paths
    assert chunks[0].chunk_id == "langgraph/persistence::000"
    assert chunks[-1].text.startswith("Persistence > Checkpointers > Memory saver")


def test_code_block_never_split_even_when_oversized() -> None:
    code = "```python\n" + "\n".join(f"x_{i} = {i}" for i in range(300)) + "\n```"
    text = f"## Big\n\n{_prose(100)}\n\n{code}\n\n{_prose(100, 'beta')}\n"
    chunks = chunk_page("p", "P", text, CFG, set())
    assert all(_fence_lines_balanced(c.text) for c in chunks)
    holding = [c for c in chunks if "x_0 = 0" in c.text]
    assert len(holding) == 1 and "x_299 = 299" in holding[0].text


def test_size_split_with_overlap() -> None:
    paras = "\n\n".join(_prose(5, f"w{i}") for i in range(30))
    chunks = chunk_page("p", "P", f"## S\n\n{paras}\n", CFG, set())
    assert len(chunks) > 2
    assert all(c.n_tokens <= CFG.max_tokens + 30 for c in chunks)
    first_tail = chunks[0].text.split("\n\n")[-1]
    assert first_tail in chunks[1].text  # overlap carried


def test_headings_inside_code_are_not_headings() -> None:
    text = "## Real\n\n```bash\n# not a heading\necho hi\n```\n"
    chunks = chunk_page("p", "P", text, CFG, set())
    assert [c.heading_path for c in chunks] == [["P", "Real"]]


def test_symbols_and_links_per_chunk() -> None:
    text = (
        "## Use\n\nCall `graph.invoke()` with `thread_id`; see `True` and "
        "[persistence](/oss/langgraph/persistence).\n\n"
        "```python\nfrom langgraph.checkpoint.memory import InMemorySaver, MemorySaver as M\n```\n"
    )
    [chunk] = chunk_page("langgraph/x", "X", text, CFG, {"langgraph/persistence"})
    assert chunk.symbols == [
        "InMemorySaver",
        "invoke",
        "MemorySaver",
        "thread_id",
    ]  # case-insensitive sort
    assert chunk.out_links == ["langgraph/persistence"]


def test_normalize_symbol() -> None:
    assert normalize_symbol("@entrypoint") == "entrypoint"
    assert normalize_symbol("langgraph.types.Command") == "Command"
    assert normalize_symbol("compile()") == "compile"
    assert normalize_symbol("pip install langgraph") is None
    assert normalize_symbol("None") is None
    assert extract_symbols("`StateGraph` and `add_node()`") == ["add_node", "StateGraph"]


def test_chunks_roundtrip(tmp_path: Path) -> None:
    chunk = Chunk("p::000", "p", ["P"], "P\n\nbody", 3, ["X"], ["q"])
    save_chunks([chunk], tmp_path / "c.jsonl")
    assert load_chunks(tmp_path / "c.jsonl") == [chunk]
    assert chunk.anchor == "p#P"
    assert count_tokens("hello world") == 2
