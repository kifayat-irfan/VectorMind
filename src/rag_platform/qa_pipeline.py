"""
Full retrieve -> rerank -> generate -> cite pipeline, with multi-turn
conversation support.

Every answer is generated strictly from retrieved chunks. Every chunk used
carries its source PDF filename and page number into the final result, so
"page 7 of handbook.pdf" is always available alongside the answer.

Two safeguards against a specific failure mode: an LLM connecting two
genuinely unrelated retrieved chunks into a false causal story (e.g.
mixing "attendance ownership" and "grade-editing permissions" — two
different features — because both chunks happened to be in the same
prompt). The system prompt explicitly forbids inferring relationships
that aren't stated within a single context block, and chunks that score
far worse than the reranker's own best match are dropped before they
ever reach the LLM, rather than trusting the model to ignore them itself.

Why a relative gap, not an absolute score floor: cross-encoder rerank
scores are not on a fixed, comparable scale across different queries — the
same model can score a genuinely relevant chunk at -7.6 for one question
and an irrelevant chunk at -8.2 for a different question. An absolute
cutoff (e.g. "drop anything below -2.0") either lets real hallucination
cases through or wrongly rejects a question's only good match, depending
on where that query's whole score distribution happens to sit. Measuring
each chunk against *that same query's* top-ranked result is stable
regardless of where the absolute scale lands.

Multi-turn memory: a bare follow-up question like "how do I fix it?" embeds
and retrieves badly on its own — the embedding model has no idea what "it"
refers to. Before retrieval, a short conversation history (if any) is used
to rewrite the question into a standalone one that carries its own context
("how do I fix a room conflict in the timetable?"). This is the standard
"condense question" pattern: one extra, cheap LLM call, only made when
there's actual history to draw on.

Only the single immediately-preceding turn is included in the final
generation prompt (not several) — real testing showed that a 2-turn window
let the model blend two genuinely unrelated topics into one answer (e.g.
mentioning a much-earlier "room conflict" question while answering a new
one about CSV imports), because the second-most-recent turn just happened
to be about something else entirely. One turn is enough for legitimate
pronoun/context resolution ("it", "that", "what about X") without dragging
in whatever else the conversation covered several messages ago.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rag_platform.llm import OllamaClient
from rag_platform.reranker import CrossEncoderReranker
from rag_platform.retriever import HybridRetriever

SYSTEM_PROMPT = (
    "You are an assistant that answers questions using ONLY the provided context. "
    "If the context does not contain the answer, say so plainly instead of guessing. "
    "Each numbered context block [n] is an independent excerpt — treat blocks as "
    "unrelated to each other unless a single block explicitly connects them. "
    "Never infer a cause, relationship, or workaround that isn't directly stated in "
    "the context, even if it seems like an obvious or reasonable conclusion — if the "
    "steps or explanation aren't written in the context, say the documentation doesn't "
    "cover it rather than reasoning your way to an answer. "
    "Cite sources inline using [n] markers that map to the numbered context blocks. "
    "\n\n"
    "FORMAT REQUIREMENTS (CRITICAL — FOLLOW EXACTLY):\n"
    "- Start with a top-level heading (# Title) that captures the main question\n"
    "- Use ## (H2) for major sections\n"
    "- Use ### (H3) for subsections if needed\n"
    "- Use **bold** for key terms, important names, and emphasized concepts\n"
    "- Use bullet lists (- item) for unordered lists\n"
    "- Use numbered lists (1. step) for sequential steps\n"
    "- End with a ## Summary section that synthesizes the key takeaways\n"
    "- Use markdown formatting consistently throughout the response"
)

CONDENSE_SYSTEM_PROMPT = (
    "You are a question rewriter. Your task is to rewrite follow-up questions "
    "as standalone questions that include any necessary context from the conversation history. "
    "This is needed because the embedding model needs a complete, self-contained question to "
    "perform accurate semantic search.\n\n"
    "STRICT FORMAT REQUIREMENTS:\n"
    "1. Output ONLY the rewritten question — NO preamble, NO quotes, NO explanation, NO prefix\n"
    "2. Do NOT say 'Sure!', 'Here is the rewritten question:', or any similar introduction\n"
    "3. Do NOT wrap the output in quotes, brackets, or any formatting\n"
    "4. If the question is already standalone, return it EXACTLY as-is, unchanged\n"
    "5. If it's a follow-up (uses 'it', 'this', 'that', 'how do I fix it'), expand it with context from history\n"
    "6. Preserve the original language of the question (English stays English, etc.)\n"
    "7. Keep the rewritten question concise — don't add unnecessary words\n\n"
    "EXAMPLES:\n"
    "- History: Q='What is the attendance workflow?' A='Teachers mark attendance...'\n"
    "  Follow-up: 'How do they do it?'\n"
    "  Rewritten: 'How do teachers mark attendance?'\n\n"
    "- History: Q='Tell me about grading' A='Teachers can edit grades...'\n"
    "  Follow-up: 'What about the final exam?'\n"
    "  Rewritten: 'What about the final exam grades?'\n\n"
    "- Standalone: 'What is chunk_size?'\n"
    "  Output: 'What is chunk_size?' (unchanged)\n\n"
    "Now rewrite the follow-up question. Output ONLY the question text."
)


DEFAULT_MAX_SCORE_GAP = 6.0
DEFAULT_HISTORY_TURNS_FOR_CONDENSING = 3
DEFAULT_HISTORY_TURNS_FOR_GENERATION = 1

ConversationTurn = tuple[str, str]


@dataclass
class Citation:
    marker: int
    source_path: str
    page_number: Optional[int]
    chunk_text: str
    score: float


@dataclass
class QAResult:
    question: str
    answer: str
    citations: list
    standalone_question: Optional[str] = None


class QAPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[CrossEncoderReranker],
        llm_client: OllamaClient,
        rerank_top_k: int = 5,
        max_score_gap: Optional[float] = DEFAULT_MAX_SCORE_GAP,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.rerank_top_k = rerank_top_k
        self.max_score_gap = max_score_gap

    async def answer(
        self,
        question: str,
        where: Optional[dict[str, Any]] = None,
        conversation_history: Optional[list[ConversationTurn]] = None,
    ) -> QAResult:
        history = conversation_history or []
        standalone_question = question
        if history:
            standalone_question = await self._condense_question(question, history)

        candidates = self.retriever.search(standalone_question, where=where)

        if self.reranker is not None:
            ranked = self.reranker.rerank(standalone_question, candidates, top_k=self.rerank_top_k)
            if ranked and self.max_score_gap is not None:
                top_score = ranked[0].get("rerank_score", 0.0)
                floor = top_score - self.max_score_gap
                ranked = [r for r in ranked if r.get("rerank_score", 0.0) >= floor]
        else:
            ranked = candidates[: self.rerank_top_k]

        if not ranked:
            return QAResult(
                question=question,
                answer="I couldn't find any sufficiently relevant indexed content to answer this question.",
                citations=[],
                standalone_question=standalone_question if standalone_question != question else None,
            )

        context_blocks = []
        citations: list[Citation] = []
        for i, hit in enumerate(ranked, start=1):
            meta = hit.get("metadata", {})
            source = meta.get("source_path", "unknown")
            page = meta.get("page_number")
            page_label = f", page {page}" if page is not None else ""
            context_blocks.append(f"[{i}] (source: {source}{page_label})\n{hit['document']}")
            citations.append(
                Citation(
                    marker=i,
                    source_path=source,
                    page_number=page,
                    chunk_text=hit["document"],
                    score=hit.get("rerank_score", hit.get("rrf_score", 0.0)),
                )
            )

        prompt_parts = []
        recent_history = history[-DEFAULT_HISTORY_TURNS_FOR_GENERATION:]
        if recent_history:
            history_text = "\n\n".join(f"Previous Q: {q}\nPrevious A: {a}" for q, a in recent_history)
            prompt_parts.append(f"Recent conversation (for context only, not a source of facts):\n{history_text}\n")

        prompt_parts.append(f"Context:\n\n{chr(10).join(context_blocks)}")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer using only the numbered context blocks above, with [n] citation markers.")
        prompt = "\n\n".join(prompt_parts)

        answer_text = await self.llm_client.generate(prompt, system=SYSTEM_PROMPT)
        return QAResult(
            question=question,
            answer=answer_text,
            citations=citations,
            standalone_question=standalone_question if standalone_question != question else None,
        )

    async def _condense_question(self, question: str, history: list[ConversationTurn]) -> str:
        recent = history[-DEFAULT_HISTORY_TURNS_FOR_CONDENSING:]
        history_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in recent)
        prompt = f"Conversation history:\n{history_text}\n\nFollow-up question: {question}\n\nStandalone question:"

        try:
            rewritten = await self.llm_client.generate(prompt, system=CONDENSE_SYSTEM_PROMPT, temperature=0.0)
        except Exception:
            return question

        rewritten = rewritten.strip().strip('"')
        return rewritten if rewritten else question
