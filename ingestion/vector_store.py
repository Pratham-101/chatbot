import numpy as np

# Patch numpy attributes removed in numpy 2.0 for compatibility with chromadb
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'uint'):
    np.uint = np.uint32

import chromadb
from chromadb import PersistentClient
from typing import List, Dict

class VectorStore:
    def __init__(self, persist_directory: str = "vector_store"):
        # Initialize the Chroma client with a persistent directory
        self.client = PersistentClient(path=persist_directory)

        # Create or get collection (no need to specify 'topic' or other metadata fields)
        self.collection = self.client.get_or_create_collection(
            name="mutual_fund_factsheets"
        )

    def add_documents(self, documents: List[Dict]):
        """Add processed documents to the vector store"""
        ids = [str(hash(doc["text"])) for doc in documents]
        embeddings = [doc["embedding"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [{"source": doc.get("source", "unknown")} for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def query(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Query the vector store for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return [
            {
                "text": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
