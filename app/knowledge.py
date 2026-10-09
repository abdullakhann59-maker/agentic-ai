"""The vector database layer (ChromaDB through LangChain).

Three collections:
  knowledge : chat examples, task playbooks and document chunks (tagged by "type")
  memory    : facts the user asked the agent to remember
  tools     : one entry per tool group, used to pick relevant tools on the direct path

The user's command is embedded ONCE per request and that vector is reused for every
search below. That keeps retrieval cheap (one embedding call instead of five).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.doc_text import read_document
from app.llm import get_embeddings

COSINE = {"hnsw": {"space": "cosine"}}


@dataclass
class Hit:
    text: str
    score: float          # cosine similarity, 0..1 (higher = more similar)
    meta: dict


@dataclass
class Retrieval:
    vector: list[float]
    chat: list[Hit] = field(default_factory=list)
    tasks: list[Hit] = field(default_factory=list)
    docs: list[Hit] = field(default_factory=list)
    memory: list[Hit] = field(default_factory=list)


class KnowledgeBase:
    def __init__(self) -> None:
        s = get_settings()
        emb = get_embeddings()
        kw = dict(embedding_function=emb, persist_directory=str(s.chroma_dir),
                  collection_configuration=COSINE)
        self.knowledge = Chroma(collection_name="knowledge", **kw)
        self.memory = Chroma(collection_name="memory", **kw)
        self.tools = Chroma(collection_name="tools", **kw)
        self.embeddings = emb
        self._lock = threading.Lock()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

    # ------------------------------------------------------------------ add
    def add_chat(self, question: str, answer: str) -> int:
        """Embed the QUESTION (that is what future users will type); store the answer as metadata."""
        doc_id = "chat_" + _hash(question.lower().strip())
        self.knowledge.add_texts([normalize_for_search(question)], metadatas=[{
            "type": "chat", "question": question, "answer": answer, "title": question[:80],
            "added_at": time.time()}],
            ids=[doc_id])
        return 1

    def add_task(self, title: str, description: str, tools: list[str], examples: list[str]) -> int:
        """Embed the title + example commands; store the full steps as metadata.

        Users type commands, not descriptions, so matching against example commands
        retrieves the right playbook more reliably than matching the long description.
        """
        embed_text = normalize_for_search(title + "\n" + "\n".join(examples))
        doc_id = "task_" + _hash(title.lower().strip())
        self.knowledge.add_texts([embed_text], metadatas=[{
            "type": "task", "title": title, "description": description,
            "tools": ",".join(tools), "examples": json.dumps(examples), "added_at": time.time()}],
            ids=[doc_id])
        return 1

    def add_document(self, path: str | Path, title: str | None = None) -> int:
        p = Path(path)
        pages = read_document(p)
        file_hash = _hash(p.read_bytes().hex())
        # skip files already ingested with identical content
        existing = self.knowledge.get(where={"file_hash": file_hash}, limit=1)
        if existing and existing.get("ids"):
            return 0
        texts, metas = [], []
        for page_no, text in pages:
            for i, chunk in enumerate(self.splitter.split_text(text)):
                if len(chunk.strip()) < 20:
                    continue
                texts.append(chunk)
                metas.append({"type": "document", "source": title or p.name, "page": page_no,
                              "chunk": i, "file_hash": file_hash, "added_at": time.time()})
        if texts:
            ids = [f"doc_{file_hash}_{m['page']}_{m['chunk']}" for m in metas]
            self.knowledge.add_texts(texts, metadatas=metas, ids=ids)
        return len(texts)

    def add_text_document(self, title: str, text: str) -> int:
        texts = [c for c in self.splitter.split_text(text) if len(c.strip()) >= 20]
        h = _hash(title + text)
        metas = [{"type": "document", "source": title, "page": 1, "chunk": i, "file_hash": h,
                  "added_at": time.time()} for i in range(len(texts))]
        if texts:
            self.knowledge.add_texts(texts, metadatas=metas, ids=[f"doc_{h}_{i}" for i in range(len(texts))])
        return len(texts)

    # --------------------------------------------------------------- search
    def retrieve(self, query: str) -> Retrieval:
        vector = self.embeddings.embed_query(normalize_for_search(query))
        r = Retrieval(vector=vector)
        r.chat = self._search(self.knowledge, vector, 2, {"type": "chat"})
        r.tasks = self._search(self.knowledge, vector, 2, {"type": "task"})
        r.docs = self._search(self.knowledge, vector, 4, {"type": "document"})
        r.memory = self._search(self.memory, vector, 3, None)
        return r

    @staticmethod
    def _search(store: Chroma, vector: list[float], k: int, where: dict | None) -> list[Hit]:
        try:
            results = store.similarity_search_by_vector_with_relevance_scores(vector, k=k, filter=where)
        except Exception:
            return []  # empty collection or filter with no matches
        # with cosine space, Chroma returns a distance; similarity = 1 - distance
        return [Hit(text=d.page_content, score=round(1 - dist, 4), meta=d.metadata) for d, dist in results]

    # ---------------------------------------------------------------- tools
    def sync_tool_groups(self, groups: dict[str, str]) -> None:
        """Index tool-group descriptions (idempotent: same ids are overwritten)."""
        names = list(groups)
        self.tools.add_texts([groups[n] for n in names], metadatas=[{"group": n} for n in names], ids=names)

    def select_tool_groups(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        hits = self._search(self.tools, vector, k, None)
        return [(h.meta["group"], h.score) for h in hits]

    # --------------------------------------------------------------- memory
    def memory_save(self, fact: str) -> None:
        self.memory.add_texts([fact], metadatas=[{"saved_at": time.time()}], ids=["mem_" + _hash(fact)])

    def memory_search(self, query: str, k: int = 5) -> list[Hit]:
        return self._search(self.memory, self.embeddings.embed_query(query), k, None)

    # --------------------------------------------------------------- counts
    def counts(self) -> dict:
        out = {}
        for t in ("chat", "task", "document"):
            res = self.knowledge.get(where={"type": t}, include=[])
            out[t] = len(res.get("ids", []))
        out["memory"] = len(self.memory.get(include=[]).get("ids", []))
        return out


_URL = re.compile(r"https?://\S+|www\.\S+")
_FILE = re.compile(r"\b[\w./-]+\.(csv|xlsx|xls|json|pdf|docx|txt|md|png|jpe?g|tiff?)\b", re.I)
_EMAIL = re.compile(r"\b[^@\s]+@[^@\s]+\.\w+\b")


def normalize_for_search(text: str) -> str:
    """Replace URLs, file names and emails with generic words before embedding.

    "scrape prices from http://shop.com/x?id=9" and "scrape prices from https://abc.in" should
    look the same to the vector search: the task matters, not the specific address.
    """
    text = _URL.sub(" website url ", text)
    text = _EMAIL.sub(" email address ", text)
    text = _FILE.sub(lambda m: f" {m.group(1).lower()} file ", text)
    return " ".join(text.split())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


_kb: KnowledgeBase | None = None
_kb_lock = threading.Lock()


def get_kb() -> KnowledgeBase:
    global _kb
    with _kb_lock:
        if _kb is None:
            _kb = KnowledgeBase()
        return _kb


def reset_kb() -> None:
    """Tests only: rebuild the knowledge base (e.g. after swapping the embedding model)."""
    global _kb
    _kb = None


def hits_to_documents(hits: list[Hit]) -> list[Document]:
    return [Document(page_content=h.text, metadata=h.meta | {"score": h.score}) for h in hits]
