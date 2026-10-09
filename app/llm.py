"""LLM and embedding model factories.

Everything that talks to Ollama goes through here, so tests can swap in fake models
with `set_overrides()` and nothing else in the code has to change.
"""
from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import get_settings

_overrides: dict[str, Any] = {}


def set_overrides(chat: BaseChatModel | None = None, embeddings: Embeddings | None = None) -> None:
    """Used by tests to replace Ollama with fake models."""
    _overrides.clear()
    if chat is not None:
        _overrides["chat"] = chat
    if embeddings is not None:
        _overrides["embeddings"] = embeddings


def get_chat_model(json_mode: bool = False) -> BaseChatModel:
    if "chat" in _overrides:
        return _overrides["chat"]
    from langchain_ollama import ChatOllama

    s = get_settings()
    kwargs: dict[str, Any] = dict(
        model=s.llm_model,
        base_url=s.ollama_base_url,
        temperature=s.llm_temperature,
        num_ctx=s.llm_num_ctx,
    )
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


_embeddings_cache: Embeddings | None = None


def get_embeddings() -> Embeddings:
    global _embeddings_cache
    if "embeddings" in _overrides:
        return _overrides["embeddings"]
    if _embeddings_cache is None:
        from langchain_ollama import OllamaEmbeddings

        s = get_settings()
        _embeddings_cache = OllamaEmbeddings(model=s.embed_model, base_url=s.ollama_base_url)
    return _embeddings_cache


def token_usage(message: Any) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) reported by the model for one response."""
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)
