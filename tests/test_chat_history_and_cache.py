from __future__ import annotations

from pathlib import Path

import pytest

from rag_platform.cache import cache_key, cached_call, get_cache
from rag_platform.chat_history import ChatHistoryStore


@pytest.fixture
def chat_store(tmp_path: Path) -> ChatHistoryStore:
    return ChatHistoryStore(tmp_path / "chat.sqlite3")


def test_chat_history_company_isolation(chat_store: ChatHistoryStore):
    chat_store.add("gitlab", "q1", "a1", [])
    chat_store.add("gitlab", "q2", "a2", [])
    chat_store.add("basecamp", "q3", "a3", [])

    assert len(chat_store.list_for_company("gitlab")) == 2
    assert len(chat_store.list_for_company("basecamp")) == 1


def test_chat_history_search(chat_store: ChatHistoryStore):
    chat_store.add("gitlab", "What is the security policy?", "answer about security", [])
    chat_store.add("gitlab", "What is the sales process?", "answer about sales", [])

    results = chat_store.list_for_company("gitlab", search="security")
    assert len(results) == 1
    assert "security" in results[0].question.lower()


def test_chat_history_star_and_delete(chat_store: ChatHistoryStore):
    entry_id = chat_store.add("gitlab", "q", "a", [])

    chat_store.toggle_star(entry_id, "gitlab")
    entries = chat_store.list_for_company("gitlab")
    assert entries[0].starred is True

    chat_store.toggle_star(entry_id, "gitlab")
    entries = chat_store.list_for_company("gitlab")
    assert entries[0].starred is False

    chat_store.delete(entry_id, "gitlab")
    assert len(chat_store.list_for_company("gitlab")) == 0


def test_chat_history_cross_tenant_mutation_blocked(chat_store: ChatHistoryStore):
    """H7 regression: one tenant must not star/delete another's entries."""
    entry_id = chat_store.add("gitlab", "q", "a", [])

    assert chat_store.toggle_star(entry_id, "basecamp") is False
    assert chat_store.list_for_company("gitlab")[0].starred is False

    assert chat_store.delete(entry_id, "basecamp") is False
    assert len(chat_store.list_for_company("gitlab")) == 1


def test_chat_history_persists_citations(chat_store: ChatHistoryStore):
    citations = [{"source": "handbook.pdf", "page": 3}]
    chat_store.add("gitlab", "q", "a", citations)
    entries = chat_store.list_for_company("gitlab")
    assert entries[0].citations == citations


def test_cache_hit_avoids_recompute():
    cache = get_cache(maxsize=10, ttl_seconds=60)
    calls = {"count": 0}

    def expensive():
        calls["count"] += 1
        return f"result_{calls['count']}"

    key = cache_key("gitlab", "some question", 5)
    v1, hit1 = cached_call(cache, key, expensive)
    v2, hit2 = cached_call(cache, key, expensive)

    assert hit1 is False
    assert hit2 is True
    assert v1 == v2
    assert calls["count"] == 1


def test_cache_key_differs_for_different_inputs():
    k1 = cache_key("gitlab", "question a", 5)
    k2 = cache_key("gitlab", "question b", 5)
    k3 = cache_key("basecamp", "question a", 5)

    assert k1 != k2
    assert k1 != k3
