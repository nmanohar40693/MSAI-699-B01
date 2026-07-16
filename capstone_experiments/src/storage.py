import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ExperimentResultStorage:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def save_run(self, strategy_name: str, config: dict, steps: list, summary_stats: dict) -> str:
        """Saves detailed experiment run data as a serialized JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{strategy_name.lower().replace(' ', '_')}_{timestamp}.json"
        full_path = os.path.join(self.results_dir, filename)

        payload = {
            "strategy": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "summary_statistics": summary_stats,
            "detailed_steps": steps
        }

        with open(full_path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Successfully archived experimental results to: {full_path}")
        return full_path
