import os
import re
import json
import logging
import hashlib
import numpy as np

logger = logging.getLogger(__name__)

# Lazy load sentence-transformers to avoid startup overhead if in mock mode
_model_instance = None
_current_model_name = None

def get_embedding_model(model_name: str):
    global _model_instance, _current_model_name
    if _model_instance is None or _current_model_name != model_name:
        logger.info(f"Loading SentenceTransformer model: {model_name}...")
        from sentence_transformers import SentenceTransformer
        _model_instance = SentenceTransformer(model_name)
        _current_model_name = model_name
    return _model_instance


class RAGIndexer:
    def __init__(self, dataset_dir: str, embedding_model_name: str = "all-MiniLM-L6-v2", cache_dir: str = None):
        self.dataset_dir = dataset_dir
        self.model_name = embedding_model_name
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(dataset_dir), "cache", "embeddings")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.chunks = []
        self.embeddings = None

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
        """Splits text into chunks of specified size and overlap, keeping lines intact where possible."""
        if not text:
            return []
        
        # Simple sliding character window
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= text_len:
                break
            start += (chunk_size - chunk_overlap)
            
        return chunks

    def build_index_for_version(self, version_tag: str, chunk_size: int = 500, chunk_overlap: int = 100):
        """Loads version files, chunks them, computes embeddings, and indexes them in memory."""
        self.chunks = []
        version_dir = os.path.join(self.dataset_dir, "versions", version_tag)
        
        if not os.path.exists(version_dir):
            raise FileNotFoundError(f"Version directory not found at: {version_dir}")

        # Find version_meta.json
        meta_file = os.path.join(version_dir, "version_meta.json")
        if not os.path.exists(meta_file):
            logger.warning(f"Metadata file missing for version {version_tag}")

        # Scan for Java code, tests, and documentation files
        subdirs = ["source_code", "tests", "documentation", "build_configs"]
        raw_files = []
        
        for sub in subdirs:
            sub_path = os.path.join(version_dir, sub)
            if not os.path.exists(sub_path):
                continue
            for root, _, files in os.walk(sub_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, version_dir)
                    raw_files.append((full_path, rel_path, sub))

        if not raw_files:
            logger.warning(f"No files found to index for version: {version_tag}")
            return

        logger.info(f"Indexing {len(raw_files)} files for version {version_tag}...")
        
        all_texts = []
        chunk_metadata_list = []
        
        for full_path, rel_path, category in raw_files:
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # Check cache via content hash and model name
                content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                clean_model_name = self.model_name.replace("/", "_")
                cache_file = os.path.join(self.cache_dir, f"{content_sha}_{clean_model_name}_{chunk_size}_{chunk_overlap}.json")

                
                file_chunks = []
                file_embeddings = []
                
                if os.path.exists(cache_file):
                    # Load from cache
                    with open(cache_file, "r") as cf:
                        cached_data = json.load(cf)
                        file_chunks = cached_data["chunks"]
                        file_embeddings = cached_data["embeddings"]
                else:
                    # Generate new chunks
                    file_chunks = self.chunk_text(content, chunk_size, chunk_overlap)
                    if file_chunks:
                        # Compute embeddings
                        model = get_embedding_model(self.model_name)
                        file_embeddings = model.encode(file_chunks).tolist()
                        
                        # Save to cache
                        with open(cache_file, "w") as cf:
                            json.dump({
                                "chunks": file_chunks,
                                "embeddings": file_embeddings
                            }, cf)

                for i, chunk in enumerate(file_chunks):
                    self.chunks.append({
                        "text": chunk,
                        "artifact_id": f"versions/{version_tag}/{rel_path}",
                        "source_file": rel_path,
                        "category": category,
                        "chunk_index": i
                    })
                    all_texts.append(chunk)
                    chunk_metadata_list.append(file_embeddings[i])
                    
            except Exception as e:
                logger.error(f"Error indexing file {rel_path}: {e}")

        if chunk_metadata_list:
            self.embeddings = np.array(chunk_metadata_list, dtype=np.float32)
            logger.info(f"Successfully constructed vector index with {len(self.chunks)} total chunks.")
        else:
            self.embeddings = np.empty((0, 0), dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> list:
        """Searches the vector index using cosine similarity."""
        if self.embeddings is None or self.embeddings.size == 0 or not self.chunks:
            logger.warning("Search query received on empty vector index.")
            return []

        # Encode query
        model = get_embedding_model(self.model_name)
        query_vector = model.encode([query])[0]
        
        # Compute Cosine Similarity
        # cosine = (A . B) / (||A|| * ||B||)
        dot_products = np.dot(self.embeddings, query_vector)
        norm_matrix = np.linalg.norm(self.embeddings, axis=1)
        norm_query = np.linalg.norm(query_vector)
        
        # Avoid division by zero
        norms = norm_matrix * norm_query
        norms[norms == 0.0] = 1e-10
        
        similarities = dot_products / norms
        
        # Get Top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            chunk_info = self.chunks[idx].copy()
            chunk_info["similarity_score"] = score
            results.append(chunk_info)
            
        return results
