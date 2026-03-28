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


def _score_document(question_terms: Counter, content: str) -> float:
    if not content:
        return 0.0

    doc_terms = Counter(_tokenize(content))
    cosine = _cosine_similarity(question_terms, doc_terms)

    # Reward exact phrase overlap with the question so short documents still rank well.
    phrase_bonus = 0.0
    lowered = content.lower()
    for token in question_terms:
        if token in lowered:
            phrase_bonus += 0.02

    return cosine + min(0.2, phrase_bonus)


def _summarize_context(question: str, passages: list[str]) -> str:
    question_terms = set(_tokenize(question))
    if not passages:
        return "I don't know based on the provided data."

    scored_sentences: list[tuple[float, str]] = []
    for passage in passages:
        passage = _clean_passage_text(passage)
        for sentence in _sentence_split(passage):
            tokens = Counter(_tokenize(sentence))
            if not tokens:
                continue
            if len(tokens) < 6:
                continue
            score = _cosine_similarity(Counter(question_terms), tokens)
            if score > 0:
                scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen = set()
    for _, sentence in scored_sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(sentence)
        if len(selected) == 3:
            break

    if not selected:
        fallback = passages[0].strip()
        fallback = _clean_passage_text(fallback)
        return f"I found relevant context, but not a strong sentence match. The document says: {fallback[:700]}"

    if len(selected) == 1:
        return selected[0]

    if len(selected) == 2:
        return f"{selected[0]} {selected[1]}"

    return f"{selected[0]} {selected[1]} {selected[2]}"


def ask(question: str, top_k: int = 3) -> tuple[str, list[str]]:
    docs = _load_documents()
    question = question.strip()
    if not question:
        return "Please enter a question.", []
    if not docs:
        return "I don't know based on the provided data.", []

    question_terms = Counter(_tokenize(question))
    ranked: list[tuple[float, dict]] = []

    for doc in docs:
        content = doc.get("page_content", "")
        score = _score_document(question_terms, content)
        if score > 0:
            ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[:top_k]
    if not selected:
        return "I don't know based on the provided data.", []

    passages = [doc.get("page_content", "") for _, doc in selected]
    answer = _summarize_context(question, passages)
    sources = [_format_source(doc.get("metadata", {}), score) for score, doc in selected]
    return answer, sources
