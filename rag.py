from __future__ import annotations

import math
import pickle
import re
import sys
import types
from collections import Counter
from functools import lru_cache
from pathlib import Path


INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"
PICKLE_PATH = INDEX_DIR / "index.pkl"

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_DEFINITION_WORDS = {
    "architecture",
    "approach",
    "concept",
    "framework",
    "method",
    "paradigm",
    "principle",
    "technique",
    "technology",
}

_INTRO_SECTION_MARKERS = (
    "introduction",
    "overview",
    "core principle",
    "architecture overview",
    "why rag",
)


def _install_pickle_stubs() -> None:
    """Install lightweight stub classes so the legacy FAISS pickle can load."""

    module_names = [
        "langchain_community",
        "langchain_community.vectorstores",
        "langchain_community.vectorstores.faiss",
        "langchain_community.docstore",
        "langchain_community.docstore.in_memory",
        "langchain_core",
        "langchain_core.documents",
        "langchain_core.documents.base",
    ]
    for name in module_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    class _Document:
        def __setstate__(self, state):
            self.__dict__.update(state)

    class _InMemoryDocstore:
        def __setstate__(self, state):
            self.__dict__.update(state)

    class _FAISS:
        def __setstate__(self, state):
            self.__dict__.update(state)

    sys.modules["langchain_core.documents"].Document = _Document
    sys.modules["langchain_core.documents.base"].Document = _Document
    sys.modules["langchain_community.docstore.in_memory"].InMemoryDocstore = _InMemoryDocstore
    sys.modules["langchain_community.vectorstores.faiss"].FAISS = _FAISS


def _unwrap_state(obj):
    state = getattr(obj, "__dict__", {})
    if isinstance(state, dict) and isinstance(state.get("__dict__"), dict):
        state = state["__dict__"]
    return state


def _normalize_document(doc) -> dict:
    if isinstance(doc, dict):
        content = doc.get("page_content", "") or ""
        metadata = doc.get("metadata", {}) or {}
    else:
        state = _unwrap_state(doc)
        if isinstance(state, dict):
            content = state.get("page_content", "") or ""
            metadata = state.get("metadata", {}) or {}
        else:
            content = getattr(doc, "page_content", "") or ""
            metadata = getattr(doc, "metadata", {}) or {}

    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "page_content": str(content),
        "metadata": metadata,
    }


@lru_cache(maxsize=1)
def _load_documents() -> list[dict]:
    if not PICKLE_PATH.exists():
        return []

    _install_pickle_stubs()
    with PICKLE_PATH.open("rb") as handle:
        payload = pickle.load(handle)

    documents: list[dict] = []
    if isinstance(payload, tuple) and payload:
        docstore = payload[0]
        raw_docs = getattr(docstore, "_dict", None)
        if isinstance(raw_docs, dict):
            documents = [_normalize_document(doc) for doc in raw_docs.values()]
    elif isinstance(payload, dict):
        documents = [_normalize_document(doc) for doc in payload.values()]

    return documents


def get_document_count() -> int:
    return len(_load_documents())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_terms(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in _STOPWORDS and len(token) > 1]


def _is_definition_question(question: str) -> bool:
    lowered = question.strip().lower()
    return lowered.startswith(("what is ", "what are ", "define ", "tell me about ", "what does ", "explain "))


def _extract_subject_terms(question: str) -> list[str]:
    lowered = question.strip().lower()
    patterns = [
        r"^(?:what is|what are|define|tell me about)\s+(.*?)[\?\.!\s]*$",
        r"^what does\s+(.*?)\s+mean[\?\.!\s]*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, lowered)
        if match:
            subject = match.group(1)
            terms = [token for token in _tokenize(subject) if token not in _STOPWORDS]
            if terms:
                return terms
    return [token for token in _content_terms(question) if token not in _STOPWORDS]


def _sentence_split(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _clean_passage_text(text: str) -> str:
    if not text:
        return ""

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r"page\s+\d+\s+of\s+\d+", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"table of contents.*", line, flags=re.IGNORECASE):
            continue
        if line.lower().startswith("retrieval-augmented generation"):
            continue
        cleaned_lines.append(line)

    cleaned = " ".join(cleaned_lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _cosine_similarity(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0

    overlap = left.keys() & right.keys()
    dot = sum(left[token] * right[token] for token in overlap)
    if not dot:
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0

    return dot / (left_norm * right_norm)


def _format_source(metadata: dict, score: float) -> str:
    source = Path(metadata.get("source", "Unknown source")).name
    page = metadata.get("page_label", metadata.get("page"))
    location = f" (page {page})" if page is not None else ""
    return f"{source}{location} | relevance: {score:.3f}"


def _score_document(question: str, question_terms: Counter, content: str, metadata: dict | None = None) -> float:
    if not content:
        return 0.0

    doc_terms = Counter(_content_terms(content))
    cosine = _cosine_similarity(question_terms, doc_terms)

    # Reward exact phrase overlap with the question so short documents still rank well.
    phrase_bonus = 0.0
    lowered = content.lower()
    for token in question_terms:
        if token in lowered:
            phrase_bonus += 0.02

    if _is_definition_question(question):
        if any(marker in lowered for marker in _INTRO_SECTION_MARKERS):
            phrase_bonus += 0.16
        if re.search(r"\b(introduction|overview|definition|core principle)\b", lowered):
            phrase_bonus += 0.12
        if metadata and isinstance(metadata, dict):
            page_value = metadata.get("page_label", metadata.get("page"))
            try:
                page_num = int(page_value)
            except Exception:
                page_num = None
            if page_num is not None and page_num <= 5:
                phrase_bonus += 0.10

    return cosine + min(0.2, phrase_bonus)


def _simplify_answer_sentence(sentence: str, question: str) -> str:
    text = re.sub(r"\s+", " ", sentence).strip()
    if not text:
        return text

    lowered = text.lower()
    if _is_definition_question(question):
        if "core principle of rag" in lowered:
            match = re.search(r"the core principle of rag is[:\s]+(.+)$", text, flags=re.IGNORECASE)
            if match:
                clause = match.group(1).strip()
                clause = re.split(r"\.\s+", clause)[0].strip()
                return f"The core principle of RAG is: {clause}"

        match = re.search(r"^(.*?\b(?:is|are)\b)\s+not\s+merely\s+[^;:.]+[;:]\s*it\s+(?:is|are)\s+(.+)$", text, flags=re.IGNORECASE)
        if match:
            return f"{match.group(1).strip()} {match.group(2).strip()}"

        match = re.search(r"^(.{1,80}?\b(?:is|are)\b\s+)(.+)$", text, flags=re.IGNORECASE)
        if match:
            leading = match.group(1).strip()
            body = match.group(2).strip()
            body = re.split(r"[;:]\s+", body)[0]
            return f"{leading} {body}".strip()

    if ";" in lowered:
        text = text.split(";", 1)[0].strip()
    if ":" in text and _is_definition_question(question):
        head, tail = text.split(":", 1)
        if len(head) < 90:
            return f"{head.strip()}: {tail.strip()}".strip()
    return text


def _summarize_context(question: str, passages: list[str]) -> str:
    no_answer = "I don\u2019t know based on provided data"
    question_terms = Counter(_content_terms(question))
    subject_terms = _extract_subject_terms(question)
    definition_question = _is_definition_question(question)
    if not passages:
        return no_answer

    scored_sentences: list[tuple[float, str]] = []
    for passage in passages:
        passage = _clean_passage_text(passage)
        for sentence in _sentence_split(passage):
            tokens = Counter(_content_terms(sentence))
            if not tokens:
                continue
            if len(tokens) < 5:
                continue

            sentence_lower = sentence.lower()
            score = _cosine_similarity(question_terms, tokens)
            if subject_terms and any(term in sentence_lower for term in subject_terms):
                score += 0.12

            if definition_question:
                subject_pattern = r"(?:%s)" % "|".join(re.escape(term) for term in subject_terms) if subject_terms else r".+"
                if re.search(rf"\b{subject_pattern}\b.{{0,40}}\b(is|are|means|refers to|stands for|describes)\b", sentence_lower):
                    score += 0.28
                if re.search(rf"\b(is|are|means|refers to|stands for|describes)\b.{{0,40}}\b{subject_pattern}\b", sentence_lower):
                    score += 0.18
                if re.search(r"\b(" + "|".join(re.escape(word) for word in _DEFINITION_WORDS) + r")\b", sentence_lower):
                    score += 0.08

            if score > 0:
                scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    if definition_question:
        for _, sentence in scored_sentences:
            sentence_lower = sentence.lower()
            subject_hit = not subject_terms or any(term in sentence_lower for term in subject_terms)
            definitional_hit = bool(
                re.search(r"\b(" + "|".join(re.escape(word) for word in _DEFINITION_WORDS) + r")\b", sentence_lower)
            )
            if subject_hit and definitional_hit:
                simplified = _simplify_answer_sentence(sentence, question)
                if simplified:
                    return simplified
        return no_answer

    selected: list[str] = []
    seen = set()
    for _, sentence in scored_sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(_simplify_answer_sentence(sentence, question))
        if len(selected) == 2:
            break

    if not selected:
        return no_answer

    if len(selected) == 1:
        return selected[0]

    return f"{selected[0]} {selected[1]}"


def ask(question: str, top_k: int = 3) -> tuple[str, list[str]]:
    no_answer = "I don\u2019t know based on provided data"
    docs = _load_documents()
    question = question.strip()
    if not question:
        return "Please enter a question.", []
    if not docs:
        return no_answer, []

    question_terms = Counter(_content_terms(question))
    if not question_terms and not _is_definition_question(question):
        return no_answer, []

    ranked: list[tuple[float, dict]] = []

    for doc in docs:
        content = doc.get("page_content", "")
        score = _score_document(question, question_terms, content, doc.get("metadata", {}))
        if score > 0:
            ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if _is_definition_question(question):
        selected = ranked
    else:
        selected = ranked[:top_k]
    if not selected:
        return no_answer, []

    best_score = selected[0][0]
    if best_score < 0.12 and not _is_definition_question(question):
        return no_answer, []

    passages = [doc.get("page_content", "") for _, doc in selected]
    answer = _summarize_context(question, passages)
    if answer == no_answer:
        return no_answer, []

    sources = [_format_source(doc.get("metadata", {}), score) for score, doc in selected]
    return answer, sources
