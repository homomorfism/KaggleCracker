"""Judged generation metrics: a hand-rolled LLM-as-judge.

Why hand-rolled instead of DeepEval's GEval (the class default): this repo
already owns a stdlib Anthropic transport with temperature 0 and a disk cache
(rag/llmcall) — DeepEval would add a dependency and defaults to OpenAI-keyed
judges the team does not run. The shape is GEval's shape: a structured rubric,
a 0–1 score, a written reason, strict JSON out.

Three of the four judged metrics are implemented. Context precision is left
out deliberately: Part 2 already measures "were the retrieved chunks
relevant" against golden chunk ids, rank-aware and for free — paying a judge
to approximate a number we compute exactly would be worse measurement.

Known judge biases and what we did: position bias does not apply (single
answers are scored, never pairs); verbosity bias — the rubric says to ignore
length and score only the property; self-preference — the judge is the same
model family as the generator, which we flag in the report rather than hide
(no second-vendor key exists in this project); judge-model mismatch — the
judge is the same tier as the generator, not weaker.
"""

import json
import re

from rag.llmcall import complete

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _judge(system, user):
    reply = complete(system, user)
    match = _JSON_RE.search(reply)
    if not match:
        raise ValueError("judge returned no JSON: %r" % reply[:200])
    verdict = json.loads(match.group(0))
    score = float(verdict["score"])
    return {
        "score": max(0.0, min(1.0, score)),
        "reason": str(verdict.get("reason", "")),
    }


_FAITHFULNESS = (
    "You are grading FAITHFULNESS: is every factual claim in the answer "
    "supported by the provided context? Ignore style and length. A refusal "
    "('the corpus does not answer this') is fully faithful when the context "
    "indeed lacks the answer. Score 1.0 = every claim grounded, 0.0 = "
    "contradicts or invents facts. Reply with ONLY JSON: "
    '{"score": <0..1>, "reason": "<one sentence>"}'
)


def faithfulness(question, answer, context):
    return _judge(
        _FAITHFULNESS,
        "Question: %s\n\nContext:\n%s\n\nAnswer to grade:\n%s"
        % (question, context, answer),
    )


_RELEVANCE = (
    "You are grading ANSWER RELEVANCE: does the answer address what the "
    "question actually asked? Ignore whether it is factually correct and "
    "ignore its length — score only whether it speaks to the question. An "
    "explicit 'the corpus does not answer this' is relevant when honest. "
    'Reply with ONLY JSON: {"score": <0..1>, "reason": "<one sentence>"}'
)


def answer_relevance(question, answer):
    return _judge(_RELEVANCE, "Question: %s\n\nAnswer to grade:\n%s" % (question, answer))


_CONTEXT_RECALL = (
    "You are grading CONTEXT RECALL: does the provided context contain the "
    "facts needed to produce the reference answer? Score the fraction of the "
    "reference answer's claims that the context supports: 1.0 = everything "
    "needed is present, 0.0 = none of it is. Reply with ONLY JSON: "
    '{"score": <0..1>, "reason": "<one sentence>"}'
)


def context_recall(golden_answer, context):
    return _judge(
        _CONTEXT_RECALL,
        "Reference answer: %s\n\nContext to grade:\n%s" % (golden_answer, context),
    )


_ANSWERER = (
    "Answer the question using ONLY the provided context passages. Quote or "
    "paraphrase the context; add nothing from outside it. If the context "
    "does not contain the answer, say exactly that the corpus does not "
    "answer this question. Be concise — a few sentences."
)


def generate_answer(question, chunks):
    """The generation under test: answer from retrieved chunks, retriever
    order preserved (no lost-in-the-middle repacking — the corpus is small
    enough that k=5 chunks fit comfortably)."""
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    answer = complete(_ANSWERER, "Context:\n\n%s\n\nQuestion: %s" % (context, question))
    return answer, context
