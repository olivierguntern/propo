"""
Base vectorielle avec ChromaDB.

POURQUOI ChromaDB ?
- Simple à démarrer (en-mémoire ou persistant)
- Pas besoin d'infrastructure externe pour un MVP
- En prod : migrer vers Pinecone / Weaviate si nécessaire

Le RAG enrichit le contexte LLM avec les règles comptables pertinentes,
ce qui réduit les hallucinations et améliore la précision des réponses.
"""
import os
from typing import Any

try:
    import chromadb
    from chromadb.utils import embedding_functions
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class VectorStore:
    def __init__(self, persist_dir: str | None = None):
        if not _CHROMA_AVAILABLE:
            self._client = None
            self._collection = None
            return

        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        self._client = chromadb.PersistentClient(path=persist_dir)

        # Embeddings légers via sentence-transformers
        try:
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception:
            self._ef = embedding_functions.DefaultEmbeddingFunction()

        self._collection = self._client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=self._ef,
        )

    def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """Indexe les documents dans ChromaDB."""
        if self._collection is None:
            return

        ids = [doc["id"] for doc in documents]
        texts = [f"{doc.get('title', '')} {doc.get('content', '')}" for doc in documents]
        metadatas = [{"source": doc.get("source", ""), "title": doc.get("title", "")} for doc in documents]

        # Upsert pour éviter les doublons
        existing = set(self._collection.get()["ids"])
        new_docs = [(i, t, m) for i, t, m in zip(ids, texts, metadatas) if i not in existing]

        if new_docs:
            new_ids, new_texts, new_metas = zip(*new_docs)
            self._collection.add(
                ids=list(new_ids),
                documents=list(new_texts),
                metadatas=list(new_metas),
            )

    def query(self, text: str, n_results: int = 3) -> list[str]:
        """Retourne les passages les plus pertinents pour un texte donné."""
        if self._collection is None:
            return []
        try:
            results = self._collection.query(
                query_texts=[text],
                n_results=min(n_results, self._collection.count() or 1),
            )
            return results["documents"][0] if results["documents"] else []
        except Exception:
            return []
