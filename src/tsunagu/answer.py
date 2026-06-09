"""One answer prompt shared by every system: context only, citations, or abstain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from tsunagu.llm import LLMClient, extract_json
from tsunagu.retrievers import RetrievedChunk

INSUFFICIENT = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You answer questions about the LangGraph documentation using ONLY the context passages provided.
Rules:
1. Use only facts stated in the context. Do not use outside knowledge.
2. Each passage starts with its label in square brackets, like [langgraph/persistence#Checkpointers]. Put the label of every passage that supports your answer in "citations", copied exactly.
3. If the context does not contain the answer, set "answer" to "{INSUFFICIENT}", "citations" to [] and "abstained" to true.
4. Reply with one JSON object and nothing else:
{{"answer": "<2 to 6 sentences>", "citations": ["<label>", "..."], "abstained": false}}"""

RETRY_MESSAGE = (
    "Your reply was not a valid JSON object with keys answer, citations and abstained. "
    "Reply again with only that JSON object."
)


class AnswerPayload(BaseModel):
    answer: str
    citations: list[str] = []
    abstained: bool = False


@dataclass
class Answer:
    answer: str
    citations: list[str]
    abstained: bool
    parse_ok: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    raw: list[str] = field(default_factory=list)


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.label}]\n{c.text}" for c in chunks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    user = f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cited_pages(citations: list[str]) -> list[str]:
    """Unique page ids from `page#heading` labels, in order."""
    return list(
        dict.fromkeys(c.split("#", 1)[0].strip().strip("[]") for c in citations if c.strip())
    )


def answer_question(
    llm: LLMClient, question: str, chunks: list[RetrievedChunk], *, tag: str
) -> Answer:
    """Answer from the given chunks; retry once if the reply is not valid JSON."""
    messages = build_messages(question, chunks)
    total_in = total_out = 0
    latency = 0.0
    raws: list[str] = []
    for attempt in range(2):
        resp = llm.complete(messages, stage="answer", tag=tag if attempt == 0 else f"{tag}:retry")
        total_in += resp.prompt_tokens
        total_out += resp.completion_tokens
        latency += resp.latency_s
        raws.append(resp.text)
        try:
            payload = AnswerPayload.model_validate(extract_json(resp.text))
        except (ValueError, ValidationError):
            messages = [
                *messages,
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content": RETRY_MESSAGE},
            ]
            continue
        abstained = payload.abstained or payload.answer.strip() == INSUFFICIENT
        citations = [] if abstained else payload.citations
        return Answer(
            payload.answer, citations, abstained, True, total_in, total_out, latency, raws
        )
    return Answer(raws[-1], [], False, False, total_in, total_out, latency, raws)
