import os
import time
import json
import logging
import resource
import numpy as np
import src.rag_indexer as rag_indexer
from src.config import ExperimentConfig
from src.evaluation import EvaluationLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_pilot")

# Target models to evaluate (100% compatible, standard architectures)
MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "microsoft-codebert-base": "microsoft/codebert-base",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
}

def clean_file_path(path: str) -> str:
    """Strips the prefix folder from relative path (e.g. source_code/src/main -> src/main)."""
    for prefix in ["source_code/", "tests/", "documentation/", "build_configs/"]:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path

def calculate_metrics(retrieved_files, target_files, k=5):
    """Calculates Precision@K, Recall@K, F1, and Reciprocal Rank."""
    # Truncate retrieval list to top K
    retrieved = retrieved_files[:k]
    
    # Check binary relevance of each retrieved item
    relevance = [1 if clean_file_path(f) in target_files else 0 for f in retrieved]
    
    # Precision@K
    precision = sum(relevance) / k if k > 0 else 0.0
    
    # Recall@K
    matched_targets = set([clean_file_path(f) for f in retrieved if clean_file_path(f) in target_files])
    recall = len(matched_targets) / len(target_files) if len(target_files) > 0 else 0.0
    
    # F1-Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    # Reciprocal Rank (RR)
    rr = 0.0
    for i, rel in enumerate(relevance):
        if rel == 1:
            rr = 1.0 / (i + 1)
            break
            
    return precision, recall, f1, rr

def run_pilot():
    # Setup directories
    base_dir = "/Users/naveenmanohar/capstone_experiments"
    dataset_dir = "/Users/naveenmanohar/capstone_preprocessing/output/prepared_dataset"
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load 10 evaluation tasks
    loader = EvaluationLoader(os.path.join(base_dir, "data", "evaluation_tasks.json"))
    tasks = loader.load_tasks()
    logger.info(f"Loaded {len(tasks)} evaluation tasks.")
    
    # Target versions present in tasks
    version_tags = sorted(list(set([t["target_version"] for t in tasks])))
    
    pilot_results = {}
    
    for model_key, model_id in MODELS.items():
        logger.info(f"=== Starting Pilot for Model: {model_key} ({model_id}) ===")
        
        # Reset sentence-transformer lazy loaded cache instance
        rag_indexer._model_instance = None
        
        # Warm start: Load model before measuring any indexing or retrieval times
        logger.info("Pre-loading model to establish clean warm-start conditions...")
        rag_indexer.get_embedding_model(model_id)
        
        # Measure Indexing Time (clean of initial download/model load latency)
        start_index_time = time.perf_counter()
        
        # Instantiate Indexers for each version and pre-build them
        indexers = {}
        try:
            # We use a unique cache directory for each model to avoid dimension conflicts
            cache_dir = os.path.join(base_dir, "cache", "embeddings", model_key)
            
            for tag in version_tags:
                indexer = rag_indexer.RAGIndexer(
                    dataset_dir=dataset_dir,
                    embedding_model_name=model_id,
                    cache_dir=cache_dir
                )
                indexer.build_index_for_version(tag)
                indexers[tag] = indexer
                
            end_index_time = time.perf_counter()
            indexing_duration = end_index_time - start_index_time
            
        except Exception as e:
            logger.error(f"Failed to build index for model {model_key}: {e}")
            continue
            
        # Run Retrieval Evaluation
        task_metrics = []
        latencies = []
        
        for task in tasks:
            tag = task["target_version"]
            query = task["description"]
            targets = task["target_files"]
            
            indexer = indexers[tag]
            
            # Measure Latency (warm-start condition)
            start_retrieval = time.perf_counter()
            search_results = indexer.search(query, top_k=5)
            end_retrieval = time.perf_counter()
            
            latency = end_retrieval - start_retrieval
            latencies.append(latency)
            
            # Extract unique file paths in order of retrieval
            retrieved_files = []
            seen = set()
            for res in search_results:
                f = res["source_file"]
                if f not in seen:
                    seen.add(f)
                    retrieved_files.append(f)
            
            # Calculate quality metrics
            p, r, f1, rr = calculate_metrics(retrieved_files, targets, k=5)
            task_metrics.append({
                "task_id": task["task_id"],
                "precision": p,
                "recall": r,
                "f1": f1,
                "rr": rr,
                "latency": latency
            })
            
        # Peak memory usage
        max_rss = resource.getrusage(resource.RESOURCE_SELF if hasattr(resource, 'RESOURCE_SELF') else resource.RUSAGE_SELF).ru_maxrss
        # RSS is in bytes on macOS
        max_rss_mb = max_rss / (1024 * 1024)
        
        # Aggregated stats
        means = {
            "precision": float(np.mean([m["precision"] for m in task_metrics])),
            "recall": float(np.mean([m["recall"] for m in task_metrics])),
            "f1": float(np.mean([m["f1"] for m in task_metrics])),
            "mrr": float(np.mean([m["rr"] for m in task_metrics])),
            "latency": float(np.mean(latencies)),
            "indexing_time": float(indexing_duration),
            "peak_memory_mb": float(max_rss_mb)
        }
        
        stds = {
            "precision": float(np.std([m["precision"] for m in task_metrics])),
            "recall": float(np.std([m["recall"] for m in task_metrics])),
            "f1": float(np.std([m["f1"] for m in task_metrics])),
            "mrr": float(np.std([m["rr"] for m in task_metrics]))
        }
        
        pilot_results[model_key] = {
            "model_id": model_id,
            "means": means,
            "stds": stds,
            "task_details": task_metrics
        }
        
        logger.info(f"Model {model_key} completed. F1: {means['f1']:.4f}, MRR: {means['mrr']:.4f}")
        
    # Write output to json
    results_path = os.path.join(results_dir, "pilot_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(pilot_results, f, indent=2)
        
    # Format and print comparison table
    print_markdown_table(pilot_results)

def print_markdown_table(results):
    baseline_key = "all-MiniLM-L6-v2"
    if baseline_key not in results:
        logger.warning(f"Baseline model {baseline_key} not present in results.")
        return
        
    base = results[baseline_key]
    
    print("\n### Baseline vs. Candidate Embedding Model Pilot Results\n")
    print("| Evaluation Metric | `all-MiniLM-L6-v2` (Baseline) | `microsoft-codebert-base` | `all-mpnet-base-v2` |")
    print("| :--- | :---: | :---: | :---: |")
    
    # Helper to format stats
    def get_row(metric_name, mean_key, std_key=None, is_efficiency=False, is_percent=False):
        row = f"| **{metric_name}** | "
        
        # Baseline
        val_b = base["means"][mean_key]
        if std_key:
            std_b = base["stds"][std_key]
            row += f"{val_b:.4f} (±{std_b:.4f}) | "
        else:
            row += f"{val_b:.4f}s | " if is_efficiency and not is_percent else f"{val_b:.2f} MB | "
            
        # Candidates
        for k in ["microsoft-codebert-base", "all-mpnet-base-v2"]:
            if k not in results:
                row += "N/A | "
                continue
            res = results[k]
            val = res["means"][mean_key]
            
            if std_key:
                std = res["stds"][std_key]
                diff = val - val_b
                sign = "+" if diff >= 0 else ""
                row += f"{val:.4f} (±{std:.4f}) [$\\Delta$ {sign}{diff:.4f}] | "
            else:
                pct_diff = ((val - val_b) / val_b * 100) if val_b > 0 else 0
                sign = "+" if pct_diff >= 0 else ""
                if is_percent:
                    row += f"{val:.2f} MB [{sign}{pct_diff:.1f}%] | "
                else:
                    row += f"{val:.4f}s [{sign}{pct_diff:.1f}%] | "
        return row
        
    print(get_row("Mean Precision@5", "precision", "precision"))
    print(get_row("Mean Recall@5", "recall", "recall"))
    print(get_row("Mean F1-score", "f1", "f1"))
    print(get_row("Mean MRR", "mrr", "mrr"))
    print(get_row("Mean Retrieval Latency", "latency", is_efficiency=True))
    print(get_row("Mean Indexing Time", "indexing_time", is_efficiency=True))
    print(get_row("Peak Memory Usage", "peak_memory_mb", is_efficiency=True, is_percent=True))
    print()

if __name__ == "__main__":
    run_pilot()
