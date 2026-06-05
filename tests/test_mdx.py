from __future__ import annotations

from tsunagu.mdx import clean_mdx

RAW = """---
title: Persistence
description: Something
---

{/* Anchor stubs */}
<a id="threads" />
import Foo from '/snippets/foo.mdx';

Intro with @[`StateGraph`][StateGraph] and @[`interrupt`].

:::python
```python
from langgraph.checkpoint.memory import InMemorySaver
```
:::

:::js
```typescript
import { MemorySaver } from "@langchain/langgraph";
```
:::

<Accordion title="Why threads">
Threads keep state.
</Accordion>

<Note>
A note.
</Note>
"""


def test_title_from_front_matter() -> None:
    assert clean_mdx(RAW).title == "Persistence"


def test_js_blocks_dropped_python_kept() -> None:
    text = clean_mdx(RAW).text
    assert "InMemorySaver" in text
    assert "MemorySaver }" not in text
    assert "typescript" not in text
    assert ":::" not in text


def test_api_refs_become_code_spans() -> None:
    page = clean_mdx(RAW)
    assert "`StateGraph`" in page.text
    assert "@[" not in page.text
    assert page.api_refs == ["interrupt", "StateGraph"]  # case-insensitive sort


def test_jsx_comments_and_imports_removed() -> None:
    text = clean_mdx(RAW).text
    assert "<Note>" not in text and "</Accordion>" not in text and "<a id" not in text
    assert "{/*" not in text and "import Foo" not in text
    assert "**Why threads**" in text
    assert "Threads keep state." in text and "A note." in text


def test_title_falls_back_to_first_h1() -> None:
    assert clean_mdx("# Hello world\n\nBody\n").title == "Hello world"
