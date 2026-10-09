"""Fake models so tests run without Ollama.

HashEmbeddings: a tiny bag-of-words embedding. Texts that share words get similar
vectors, which is enough to test retrieval thresholds and routing.

ScriptedChatModel: returns pre-written replies (text or tool calls) in order, so the
agent graph can be tested step by step.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

STOP = {"the", "a", "an", "of", "to", "and", "from", "for", "in", "on", "me", "my", "is", "it", "this", "with"}


class HashEmbeddings(Embeddings):
    dim = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w in STOP:
                continue
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


Reply = AIMessage | Callable[[list[BaseMessage]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    replies: list[Any] = []
    calls: list[list[BaseMessage]] = []
    bound_tools: list[list[str]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        self.bound_tools.append([getattr(t, "name", str(t)) for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001
        self.calls.append(list(messages))
        if not self.replies:
            msg = AIMessage(content="(no scripted reply left)")
        else:
            nxt = self.replies.pop(0)
            msg = nxt(messages) if callable(nxt) else nxt
        msg = msg.model_copy()
        msg.usage_metadata = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        return ChatResult(generations=[ChatGeneration(message=msg)])


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])
