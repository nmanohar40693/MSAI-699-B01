import os
import json
import logging

logger = logging.getLogger(__name__)

class EvaluationLoader:
    def __init__(self, tasks_file_path: str):
        self.tasks_file_path = tasks_file_path
        self.tasks = []

    def load_tasks(self) -> list:
        """Loads evaluation tasks from the JSON file."""
        if not os.path.exists(self.tasks_file_path):
            raise FileNotFoundError(f"Evaluation tasks file not found at: {self.tasks_file_path}")
            
        with open(self.tasks_file_path, "r") as f:
            self.tasks = json.load(f)
            
        logger.info(f"Successfully loaded {len(self.tasks)} evaluation tasks.")
        return self.tasks

    def get_tasks_for_version(self, version: str) -> list:
        """Filters evaluation tasks for a specific version."""
        if not self.tasks:
            self.load_tasks()
        return [t for t in self.tasks if t.get("target_version") == version]
