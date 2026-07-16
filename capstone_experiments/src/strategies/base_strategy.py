import os
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseContextStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def construct_context(self, task: dict, dataset_dir: str) -> str:
        """Constructs the contextual text block to append to the LLM prompt."""
        pass


class PromptOnlyStrategy(BaseContextStrategy):
    def __init__(self):
        super().__init__("Prompt-Only")

    def construct_context(self, task: dict, dataset_dir: str) -> str:
        """Prompt-only strategy does not construct extra repository context. 
        It returns an empty string, relying purely on the default query.
        """
        return ""


class RAGStrategy(BaseContextStrategy):
    def __init__(self):
        super().__init__("RAG")
        self.indexer = None
        self.last_version = None

    def construct_context(self, task: dict, dataset_dir: str) -> str:
        """Retrieves matching code/document chunks using SentenceTransformer similarity."""
        version = task.get("target_version")
        if not version:
            return ""

        # Retrieve settings from config
        from src.config import ExperimentConfig
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(script_dir, "config", "default_config.json")
        config = ExperimentConfig(config_path)

        from src.rag_indexer import RAGIndexer
        if self.indexer is None or self.last_version != version:
            self.indexer = RAGIndexer(
                dataset_dir=dataset_dir,
                embedding_model_name=config.embedding_model_name
            )
            self.indexer.build_index_for_version(
                version_tag=version,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            self.last_version = version

        # Search for top-K matching chunks
        query = task["description"]
        top_k = config.top_k
        matched_chunks = self.indexer.search(query=query, top_k=top_k)

        # Format context block
        context_blocks = []
        for i, chunk in enumerate(matched_chunks):
            block = (
                f"--- Retrieved Chunk #{i+1} ---\n"
                f"File: {chunk['artifact_id']}\n"
                f"Category: {chunk['category']}\n"
                f"Similarity Score: {chunk['similarity_score']:.4f}\n"
                f"Content:\n{chunk['text']}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)


class MemoryAugmentedPromptingStrategy(BaseContextStrategy):
    def __init__(self):
        super().__init__("Memory-Augmented Prompting")

    def construct_context(self, task: dict, dataset_dir: str) -> str:
        """Retrieves accumulated session/project memory from prior interactions."""
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        memory_path = os.path.join(script_dir, "results", "session_memory.json")
        
        if not os.path.exists(memory_path):
            return "No prior project memory accumulated yet."
            
        try:
            with open(memory_path, "r") as f:
                memory_logs = json.load(f)
        except Exception:
            return "Error loading prior project memory."
            
        if not memory_logs:
            return "No prior project memory accumulated yet."

        context_blocks = ["=== Accumulated Project Memory from Prior Interactions ==="]
        for entry in memory_logs:
            block = (
                f"- Task {entry['task_id']}: {entry['description'][:80]}...\n"
                f"  Response Outcome Summary: {entry['response_summary']}\n"
            )
            context_blocks.append(block)
            
        context_blocks.append("=========================================================")
        return "\n".join(context_blocks)


class LifecycleGuidedContextStrategy(BaseContextStrategy):
    def __init__(self):
        super().__init__("Lifecycle-Guided Context Construction Strategy")
        self.graph_builder = None
        self.indexer = None
        self.last_version = None

    def construct_context(self, task: dict, dataset_dir: str) -> str:
        """Traverses the Lifecycle-Guided Project Knowledge Graph starting from semantic entry points."""
        version = task.get("target_version")
        if not version:
            return ""

        # Load configs
        from src.config import ExperimentConfig
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(script_dir, "config", "default_config.json")
        config = ExperimentConfig(config_path)

        # 1. Initialize and Build the Knowledge Graph for this version
        from src.graph_builder import LifecycleProjectGraph
        if self.graph_builder is None or self.last_version != version:
            self.graph_builder = LifecycleProjectGraph(dataset_dir)
            self.graph_builder.build_graph_for_version(version)
            self.last_version = version

        # 2. Get entry nodes (find semantically relevant nodes using RAG search)
        from src.rag_indexer import RAGIndexer
        if self.indexer is None or self.last_version != version:
            self.indexer = RAGIndexer(dataset_dir, config.embedding_model_name)
            self.indexer.build_index_for_version(version, config.chunk_size, config.chunk_overlap)

        # Find top 2 file chunks to serve as entry points
        query = task["description"]
        top_matches = self.indexer.search(query=query, top_k=2)
        entry_nodes = [m["artifact_id"] for m in top_matches]

        # 3. Traverse the graph to retrieve connected lifecycle artifacts (depth 2)
        traversed_nodes = self.graph_builder.traverse_for_context(entry_nodes, max_depth=2)

        # 4. Format context block
        context_blocks = ["=== Lifecycle-Guided Project Knowledge Graph Context ==="]
        for node_id, data, depth in traversed_nodes:
            # Format according to type
            node_type = data.get("type", "unknown")
            stage = data.get("lifecycle_stage", "unknown")
            name = data.get("name", "unnamed")
            
            block_header = f"Node: {node_id} (Type: {node_type}, Stage: {stage}, Relation Depth: {depth})"
            
            if node_type == "commit":
                content = f"Message: {data.get('message', '')}\nAuthor: {data.get('author', '')}\nTimestamp: {data.get('timestamp', '')}"
            elif node_type in ["issue", "pull_request"]:
                content = f"Title: {name}\nState: {data.get('state', '')}\nDescription: {data.get('text_content', '')}"
            else:
                # Code files/Documentation
                content = f"Name: {name}\nPath: {data.get('path', '')}"
                
            context_blocks.append(f"{block_header}\n{content}\n")
            
        context_blocks.append("=========================================================")
        return "\n".join(context_blocks)
