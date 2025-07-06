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

    def add_texts(self, texts: List[str], metadatas: List[Dict] = None):
        """Add texts to the vector store with automatic embedding generation
        
        Args:
            texts: List of text strings to add
            metadatas: List of metadata dictionaries (optional)
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Generate unique IDs
        ids = [f"text_{i}_{hash(text) % 1000000}" for i, text in enumerate(texts)]
        
        # Split into batches of 5000
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            print(f"Adding batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size} ({len(batch_ids)} texts)")
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )

    def add_documents(self, documents: List[Dict]):
        """Add processed documents to the vector store"""
        # Use provided IDs if available, otherwise generate unique ones
        ids = []
        embeddings = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # Use provided ID or generate a unique one
            if 'id' in doc:
                doc_id = str(doc['id'])
            else:
                doc_id = f"doc_{i}_{hash(doc['text']) % 1000000}"
            
            ids.append(doc_id)
            embeddings.append(doc["embedding"])
            texts.append(doc["text"])
            
            # Enhanced metadata
            metadata = {
                "source": doc.get("source", "unknown"),
                "fund_name": doc.get("fund_name", ""),
                "chunk_type": doc.get("chunk_type", "fund_info")
            }
            metadatas.append(metadata)

        # Split into batches of 5000 (below the limit of 5461)
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            print(f"Adding batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size} ({len(batch_ids)} documents)")
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )

    def count_documents(self) -> int:
        """Return the total number of documents in the vector store"""
        return self.collection.count()

    def query(self, query_embedding: List[float], k: int = 5, score_threshold: float = 0.0) -> List[Dict]:
        """Query the vector store for similar documents
        
        Args:
            query_embedding: The embedding vector to query with
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1) for results
            
        Returns:
            List of dicts with text, metadata, distance and score
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Convert distances to similarity scores (1 - normalized distance)
        max_dist = max(results["distances"][0]) if results["distances"][0] else 1
        scores = [1 - (dist/max_dist) for dist in results["distances"][0]]

        return [
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "score": score
            }
            for doc, meta, dist, score in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                scores
            )
            if score >= score_threshold  # Filter by score threshold
        ]
