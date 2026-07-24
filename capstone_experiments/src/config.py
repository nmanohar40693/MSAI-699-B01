import os
import json
import logging

logger = logging.getLogger(__name__)

class ExperimentConfig:
    def __init__(self, config_path: str = None):
        self.dataset_dir = ""
        self.gemini_model_name = "gemini-3.5-flash"
        self.temperature = 0.0
        self.max_output_tokens = 1024
        self.api_key = ""
        self.mock_mode = True
        
        # Embedding config defaults
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.top_k = 5
        self.chunk_size = 500
        self.chunk_overlap = 100

        # Graph optimization parameters
        self.similarity_threshold = 0.0
        self.max_depth = 2
        self.weight_tests_class = 1.0
        self.weight_resolves_issue = 1.0
        self.weight_modified_file = 1.0
        self.max_context_chars = 8000

        if config_path:
            self.load_from_json(config_path)

    def load_from_json(self, config_path: str):
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return

        with open(config_path, "r") as f:
            data = json.load(f)

        self.dataset_dir = data.get("dataset_dir", self.dataset_dir)
        self.gemini_model_name = data.get("gemini_model_name", self.gemini_model_name)
        self.temperature = data.get("temperature", self.temperature)
        self.max_output_tokens = data.get("max_output_tokens", self.max_output_tokens)
        self.api_key = data.get("api_key", self.api_key)
        self.mock_mode = data.get("mock_mode", self.mock_mode)
        
        # Load embedding settings
        self.embedding_model_name = data.get("embedding_model_name", self.embedding_model_name)
        self.top_k = data.get("top_k", self.top_k)
        self.chunk_size = data.get("chunk_size", self.chunk_size)
        self.chunk_overlap = data.get("chunk_overlap", self.chunk_overlap)

        # Load graph parameters
        self.similarity_threshold = data.get("similarity_threshold", self.similarity_threshold)
        self.max_depth = data.get("max_depth", self.max_depth)
        self.weight_tests_class = data.get("weight_tests_class", self.weight_tests_class)
        self.weight_resolves_issue = data.get("weight_resolves_issue", self.weight_resolves_issue)
        self.weight_modified_file = data.get("weight_modified_file", self.weight_modified_file)
        self.max_context_chars = data.get("max_context_chars", self.max_context_chars)
        
        logger.info(f"Loaded config successfully from {config_path}")

    def to_dict(self) -> dict:
        return {
            "dataset_dir": self.dataset_dir,
            "gemini_model_name": self.gemini_model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "mock_mode": self.mock_mode,
            "embedding_model_name": self.embedding_model_name,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "similarity_threshold": self.similarity_threshold,
            "max_depth": self.max_depth,
            "weight_tests_class": self.weight_tests_class,
            "weight_resolves_issue": self.weight_resolves_issue,
            "weight_modified_file": self.weight_modified_file,
            "max_context_chars": self.max_context_chars
        }

