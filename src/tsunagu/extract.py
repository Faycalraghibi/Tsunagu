"""Schema-guided triplet extraction: LlamaIndex SchemaLLMPathExtractor or our own JSON prompt."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from tsunagu.chunking import Chunk
from tsunagu.graph import ENTITY_TYPES, LLM_RELATIONS, ExtractedTriplet
from tsunagu.llm import LLMClient, extract_json

TYPE_BY_UPPER = {t.upper(): t for t in ENTITY_TYPES}

EXTRACT_PROMPT = """Extract a knowledge graph from this passage of the LangGraph documentation.
Entity types: Concept, Class, Function, Parameter, Feature.
Relation types: USES, REQUIRES, CONFIGURES, PART_OF, ALTERNATIVE_TO, DEPRECATED_BY, RETURNS.
Rules:
- Use names exactly as written in the passage; keep code names such as InMemorySaver unchanged.
- Only extract relations the passage states or directly shows.
- At most {max_triplets} triplets.
Reply with one JSON object only:
{{"triplets": [{{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "..."}}]}}

Passage ({heading}):
{text}"""


class Extractor(Protocol):
    def extract(self, chunk: Chunk) -> list[ExtractedTriplet]: ...


def filter_strict(items: list[dict[str, Any]], chunk_id: str) -> list[ExtractedTriplet]:
    """Keep only schema types and relations; normalise casing; drop self-loops and empties."""
    kept: list[ExtractedTriplet] = []
    for item in items:
        subject = str(item.get("subject", "")).strip().strip("`")
        obj = str(item.get("object", "")).strip().strip("`")
        s_type = TYPE_BY_UPPER.get(str(item.get("subject_type", "")).strip().upper())
        o_type = TYPE_BY_UPPER.get(str(item.get("object_type", "")).strip().upper())
        relation = str(item.get("relation", "")).strip().upper().replace(" ", "_")
        if not subject or not obj or s_type is None or o_type is None:
            continue
        if relation not in LLM_RELATIONS or subject.lower() == obj.lower():
            continue
        if len(subject) > 80 or len(obj) > 80:
            continue
        kept.append(ExtractedTriplet(subject, s_type, relation, obj, o_type, chunk_id))
    return kept


class JsonExtractor:
    """Our own schema-guided prompt (fallback backend)."""

    def __init__(self, llm: LLMClient, max_triplets: int = 10, tag: str = "json") -> None:
        self.llm = llm
        self.max_triplets = max_triplets
        self.tag = tag

    def extract(self, chunk: Chunk) -> list[ExtractedTriplet]:
        prompt = EXTRACT_PROMPT.format(
            max_triplets=self.max_triplets, heading=" > ".join(chunk.heading_path), text=chunk.text
        )
        resp = self.llm.complete(
            [{"role": "user", "content": prompt}], stage="extraction", tag=self.tag
        )
        try:
            data = extract_json(resp.text)
        except ValueError:
            return []
        items = data.get("triplets", [])
        if not isinstance(items, list):
            return []
        return filter_strict([i for i in items if isinstance(i, dict)], chunk.chunk_id)


def _cached_llama_llm(client: LLMClient, tag: str) -> Any:
    """A LlamaIndex CustomLLM that routes every completion through our cached client."""
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.llms import (
        CompletionResponse,
        CompletionResponseGen,
        CustomLLM,
        LLMMetadata,
    )
    from llama_index.core.llms.callbacks import llm_completion_callback

    class CachedLlamaLLM(CustomLLM):
        model_name: str = "tsunagu-cached"
        _client: Any = PrivateAttr()
        _tag: str = PrivateAttr()

        def __init__(self, inner: LLMClient, inner_tag: str) -> None:
            super().__init__()
            self._client = inner
            self._tag = inner_tag

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=32000,
                num_output=2048,
                model_name=self.model_name,
                is_chat_model=False,
                is_function_calling_model=False,
            )

        @llm_completion_callback()
        def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponse:
            resp = self._client.complete(
                [{"role": "user", "content": prompt}], stage="extraction", tag=self._tag
            )
            return CompletionResponse(text=resp.text)

        @llm_completion_callback()
        def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponseGen:
            response = self.complete(prompt, formatted=formatted, **kwargs)

            def gen() -> CompletionResponseGen:
                yield response

            return gen()

    return CachedLlamaLLM(client, tag)


class LlamaIndexExtractor:
    """LlamaIndex `SchemaLLMPathExtractor`, strict, uppercase schema (see Task 16 note)."""

    def __init__(self, llm: LLMClient, max_triplets: int = 10, tag: str = "llamaindex") -> None:
        from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

        upper_types = tuple(t.upper() for t in ENTITY_TYPES)
        self.extractor = SchemaLLMPathExtractor(
            llm=_cached_llama_llm(llm, tag),
            possible_entities=Literal[upper_types],  # type: ignore[valid-type,arg-type]
            possible_relations=Literal[LLM_RELATIONS],  # type: ignore[valid-type,arg-type]
            kg_validation_schema={t: list(LLM_RELATIONS) for t in upper_types},  # type: ignore[misc]
            strict=True,
            max_triplets_per_chunk=max_triplets,
            num_workers=1,
        )

    def extract(self, chunk: Chunk) -> list[ExtractedTriplet]:
        from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY
        from llama_index.core.schema import TextNode

        [node] = self.extractor([TextNode(text=chunk.text, id_=chunk.chunk_id)])
        entities = {n.id: n for n in node.metadata.get(KG_NODES_KEY, [])}
        items: list[dict[str, Any]] = []
        for rel in node.metadata.get(KG_RELATIONS_KEY, []):
            subj, obj = entities.get(rel.source_id), entities.get(rel.target_id)
            if subj is None or obj is None:
                continue
            items.append(
                {
                    "subject": subj.name,
                    "subject_type": subj.label,
                    "relation": rel.label,
                    "object": obj.name,
                    "object_type": obj.label,
                }
            )
        return filter_strict(items, chunk.chunk_id)


def build_extractor(
    backend: str, llm: LLMClient, max_triplets: int, tag: str | None = None
) -> Extractor:
    if backend == "llamaindex":
        return LlamaIndexExtractor(llm, max_triplets, tag or "llamaindex")
    if backend == "json":
        return JsonExtractor(llm, max_triplets, tag or "json")
    raise ValueError(f"unknown extraction backend {backend}")
