import logging
import numpy as np

logger = logging.getLogger(__name__)

class ExperimentMetricsCollector:
    def __init__(self):
        self.raw_records = []

    def record_step(self, task_id: str, strategy: str, latency: float, input_tokens: int, output_tokens: int, context_len: int):
        """Records metrics for an individual task execution step."""
        record = {
            "task_id": task_id,
            "strategy": strategy,
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "context_length_chars": context_len,
            # Qualitative placeholders to be filled during evaluation stage (e.g. human grading or LLM evaluation)
            "eval_accuracy": None,
            "eval_completeness": None,
            "eval_relevance": None,
            "eval_consistency": None
        }
        self.raw_records.append(record)
        logger.info(f"Recorded step for {task_id}: Latency={latency:.2f}s, InputTokens={input_tokens}, ContextChars={context_len}")

    def compute_summary_statistics(self) -> dict:
        """Calculates global statistical summaries across all recorded steps."""
        if not self.raw_records:
            return {}

        latencies = [r["latency_seconds"] for r in self.raw_records]
        input_tokens = [r["input_tokens"] if r["input_tokens"] is not None else 0 for r in self.raw_records]
        output_tokens = [r["output_tokens"] if r["output_tokens"] is not None else 0 for r in self.raw_records]
        contexts = [r["context_length_chars"] for r in self.raw_records]


        summary = {
            "total_runs": len(self.raw_records),
            "latency": {
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies)),
                "total": float(np.sum(latencies))
            },
            "tokens": {
                "total_input": int(np.sum(input_tokens)),
                "total_output": int(np.sum(output_tokens)),
                "mean_input_per_run": float(np.mean(input_tokens))
            },
            "context_length": {
                "mean_chars": float(np.mean(contexts)),
                "max_chars": int(np.max(contexts))
            }
        }
        return summary
