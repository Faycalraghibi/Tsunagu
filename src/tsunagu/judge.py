"""LLM judge: correctness (0/1/2), claim-level faithfulness, citation precision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tsunagu.answer import Answer, cited_pages, format_context
from tsunagu.llm import LLMClient, extract_json
from tsunagu.questions import Question
from tsunagu.retrievers import RetrievedChunk

CORRECTNESS_PROMPT = """You grade an answer to a question about the LangGraph documentation.

Question: {question}
Reference answer: {gold}
Candidate answer: {answer}

Rubric:
2 = correct and complete: states every key point of the reference answer and contradicts none.
1 = partially correct: at least one key point is right, but a key point is missing or slightly wrong.
0 = wrong, contradicts the reference, irrelevant, or says it cannot answer.
Extra correct detail that is not in the reference does not lower the score.

Reply with one JSON object only: {{"score": 0, "reason": "<one sentence>"}} where score is 0, 1 or 2."""

FAITHFULNESS_PROMPT = """Split the answer into atomic factual claims. For each claim, decide whether the context passages support it.

Context passages (each starts with its label in square brackets):
{context}

Answer:
{answer}

Reply with one JSON object only:
{{"claims": [{{"claim": "...", "supported": true, "support_labels": ["<label>"]}}]}}"""


@dataclass
class Judgement:
    correctness: int | None
    correctness_reason: str
    claims_total: int = 0
    claims_supported: int = 0
    cited_pages: int = 0
    precise_citations: int = 0


def judge_correctness(llm: LLMClient, question: Question, answer: str) -> tuple[int | None, str]:
    prompt = CORRECTNESS_PROMPT.format(
        question=question.question, gold=question.gold_answer, answer=answer
    )
    resp = llm.complete([{"role": "user", "content": prompt}], stage="judge", tag="correctness")
    try:
        data = extract_json(resp.text)
        score = int(data["score"])
    except (ValueError, KeyError, TypeError):
        return None, f"unparseable judge reply: {resp.text[:200]}"
    if score not in (0, 1, 2):
        return None, f"score out of range: {score}"
    return score, str(data.get("reason", ""))


def judge_faithfulness(
    llm: LLMClient, chunks: list[RetrievedChunk], answer: str
) -> list[dict[str, Any]]:
    prompt = FAITHFULNESS_PROMPT.format(context=format_context(chunks), answer=answer)
    resp = llm.complete([{"role": "user", "content": prompt}], stage="judge", tag="faithfulness")
    try:
        claims = extract_json(resp.text).get("claims", [])
    except ValueError:
        return []
    return [c for c in claims if isinstance(c, dict)] if isinstance(claims, list) else []


def citation_precision(
    citations: list[str], gold_pages: list[str], claims: list[dict[str, Any]]
) -> tuple[int, int]:
    """Return (cited pages, cited pages that are gold or support a supported claim)."""
    pages = cited_pages(citations)
    supporting = {
        str(label).strip("[] ").split("#", 1)[0]
        for claim in claims
        if claim.get("supported")
        for label in claim.get("support_labels", []) or []
    }
    precise = sum(1 for p in pages if p in gold_pages or p in supporting)
    return len(pages), precise


def judge_answer(
    llm: LLMClient, question: Question, chunks: list[RetrievedChunk], ans: Answer
) -> Judgement:
    """Score one answer. Abstention cases are decided without calling the judge."""
    if question.type == "unanswerable":
        if ans.abstained:
            return Judgement(2, "correct abstention")
        return Judgement(0, "answered an unanswerable question")
    if ans.abstained:
        return Judgement(0, "false abstention")
    score, reason = judge_correctness(llm, question, ans.answer)
    claims = judge_faithfulness(llm, chunks, ans.answer)
    supported = sum(1 for c in claims if c.get("supported") is True)
    n_cited, n_precise = citation_precision(ans.citations, question.gold_pages, claims)
    return Judgement(score, reason, len(claims), supported, n_cited, n_precise)
