# -*- coding: utf-8 -*-
"""
[DEPRECATED] Minimal RAG store using FAISS + Ollama embeddings.

⚠️  WARNING: This module is deprecated. The search functionality has been migrated to 
PostgreSQL + pgvector. This file is kept for backward compatibility with /rag/answer 
and /rag/index endpoints, but /rag/search now uses PostgreSQL + pgvector directly.

- Store dir layout:
    rag_store/
      - index.faiss
      - meta.jsonl        # one JSON per line: {"id": str, "content": str, "metadata": {...}}
      - dim.txt           # embedding dimension
- Embedding model default: bge-m3 (better for zh). Change via env OLLAMA_EMBED_MODEL.
- Ollama base default: http://127.0.0.1:11434  (env: OLLAMA_BASE)
"""

import os, json, time, base64, pathlib, math
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np

# --- FAISS import (cpu first) ---
HAS_FAISS = False
faiss = None
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    try:
        import faiss_cpu as faiss  # type: ignore
        HAS_FAISS = True
    except Exception:
        # 不拋出錯誤，允許模組載入（但 RAGStore 功能將不可用）
        print("--- [WARNING] faiss/faiss_cpu 未安裝，RAGStore 功能將不可用 ---")


def _normed(v: np.ndarray) -> np.ndarray:
    # cosine via inner product: normalize vectors
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


class RAGStore:
    def __init__(self, store_dir: str = "rag_store",
                 ollama_base: Optional[str] = None,
                 embed_model: Optional[str] = None):
        self.store_dir   = pathlib.Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_base = ollama_base or os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
        self.index_path  = self.store_dir / "index.faiss"
        self.meta_path   = self.store_dir / "meta.jsonl"
        self.dim_path    = self.store_dir / "dim.txt"

        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None
        self.meta_count: int = 0

        self._load_index()

    # ---------- Embeddings ----------
    def _embed_one(self, text: str) -> List[float]:
        url = f"{self.ollama_base.rstrip('/')}/api/embeddings"
        payload = {"model": self.embed_model, "prompt": text}
        # Newer Ollama also accepts {"input": text}
        try:
            r = requests.post(url, json=payload, timeout=120)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_base}. "
                f"Please ensure Ollama is running and accessible. "
                f"Error: {str(e)}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"Ollama request timed out after 120s. "
                f"Model '{self.embed_model}' may not be loaded. "
                f"Try: ollama pull {self.embed_model}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get embeddings from Ollama: {str(e)}"
            ) from e

        if r.status_code != 200:
            error_msg = r.text[:200] if r.text else "No error message"
            raise RuntimeError(
                f"Embedding failed [{r.status_code}]: {error_msg}. "
                f"Model '{self.embed_model}' may not be available. "
                f"Try: ollama pull {self.embed_model}"
            )
        data = r.json()
        # Accept both shapes:
        if "embedding" in data:
            return data["embedding"]
        elif "embeddings" in data and data["embeddings"]:
            return data["embeddings"][0]["embedding"]
        else:
            raise RuntimeError(f"Unexpected embedding response: {list(data.keys())}")

    def _ensure_index(self, dim: int):
        if self.index is None:
            # Inner Product + normalized = cosine similarity
            self.index = faiss.IndexFlatIP(dim)
            self.dim = dim

    def _load_index(self):
        if self.index_path.exists() and self.dim_path.exists():
            d = int(self.dim_path.read_text().strip())
            self.index = faiss.read_index(str(self.index_path))
            self.dim = d
        if self.meta_path.exists():
            # count lines for next IDs
            with self.meta_path.open("r", encoding="utf-8") as f:
                self.meta_count = sum(1 for _ in f)
        else:
            self.meta_count = 0

    # ---------- Persist ----------
    def _append_meta(self, rows: List[Dict[str, Any]]):
        with self.meta_path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _save_index(self):
        assert self.index is not None and self.dim is not None
        faiss.write_index(self.index, str(self.index_path))
        self.dim_path.write_text(str(self.dim), encoding="utf-8")

    # ---------- Public: add & search ----------
    def add_docs(self, docs: List[Dict[str, Any]]) -> int:
        """
        docs: list of {"id": str, "content": str, "metadata": dict}
        returns: added count
        """
        if not docs:
            return 0

        vecs = []
        rows = []
        for d in docs:
            content = (d.get("content") or "").strip()
            if not content:
                # skip empty content docs
                continue
            emb = self._embed_one(content)
            if self.dim is None:
                self._ensure_index(len(emb))
            elif len(emb) != self.dim:
                raise RuntimeError(f"Embedding dimension changed: got {len(emb)} != {self.dim}")
            vecs.append(emb)
            rows.append({
                "id": d.get("id", f"doc_{self.meta_count + len(rows)}"),
                "content": content,
                "metadata": d.get("metadata", {})
            })

        if not rows:
            return 0

        X = np.array(vecs, dtype="float32")
        X = _normed(X)
        assert self.index is not None
        self.index.add(X)
        self._append_meta(rows)
        self.meta_count += len(rows)
        self._save_index()
        return len(rows)

    def search(self, query: str, top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        filters: e.g. {"events_true_any": ["smoking_outside_zone","double_parking_lane_block"],
                       "video": "Video_1.avi"}
        """
        if not query.strip():
            return []
        emb = self._embed_one(query.strip())
        if self.dim is None:
            raise RuntimeError("Index empty.")
        if len(emb) != self.dim:
            raise RuntimeError(f"Query dim {len(emb)} != {self.dim}")
        q = _normed(np.array([emb], dtype="float32"))
        assert self.index is not None
        D, I = self.index.search(q, top_k * 5)  # over-fetch then filter

        # load metas
        metas = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        def pass_filter(meta: Dict[str, Any]) -> bool:
            if not filters:
                return True
            # events_true_any: at least one event is True in this doc's metadata
            if "events_true_any" in filters:
                need = set(filters["events_true_any"])
                evts = set(meta.get("events_true", []))
                if need.isdisjoint(evts):
                    return False
            if "video" in filters:
                if str(meta.get("video")) != str(filters["video"]):
                    return False
            if "time_contains" in filters:
                tfrag = str(filters["time_contains"])
                if tfrag not in str(meta.get("time_range", "")):
                    return False
            return True

        hits = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(lines):
                continue
            row = json.loads(lines[idx])
            meta = row.get("metadata", {})
            if pass_filter(meta):
                hits.append({
                    "score": float(score),
                    "id": row.get("id"),
                    "content": row.get("content"),
                    "metadata": meta
                })
            if len(hits) >= top_k:
                break
        return hits
