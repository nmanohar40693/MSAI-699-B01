import os
import sys
import json
import logging
import optuna
import numpy as np
import src.rag_indexer as rag_indexer
from src.graph_builder import LifecycleProjectGraph
from src.evaluation import EvaluationLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tune_hyperparameters")

# Set optuna logging level to warning to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def clean_file_path(path: str) -> str:
    """Strips the prefix folder from relative path (e.g. source_code/src/main -> src/main)."""
    if path.startswith("versions/"):
        parts = path.split("/", 2)
        if len(parts) > 2:
            path = parts[2]
            
    for prefix in ["source_code/", "tests/", "documentation/", "build_configs/"]:
        if path.startswith(prefix):
            return path[len(prefix):]
    return path

def traverse_with_weights(graph, entry_nodes, max_depth, weights, max_chars):
    """BFS graph traversal with edge relationship weights and character budget constraint."""
    # queue elements: (node_id, depth, score)
    queue = [(node, 0, 1.0) for node in entry_nodes if graph.has_node(node)]
    visited = {}
    for node, _, score in queue:
        visited[node] = score
        
    retrieved = []
    
    while queue:
        curr, depth, score = queue.pop(0)
        retrieved.append((curr, graph.nodes[curr], score))
        
        if depth >= max_depth:
            continue
            
        # Successors
        for n in graph.successors(curr):
            rel_type = graph[curr][n].get('relationship_type', 'unknown')
            w = weights.get(rel_type, 1.0)
            new_score = score * w / (depth + 1)
            if n not in visited or new_score > visited[n]:
                visited[n] = new_score
                queue.append((n, depth + 1, new_score))
                
        # Predecessors
        for n in graph.predecessors(curr):
            rel_type = graph[n][curr].get('relationship_type', 'unknown')
            w = weights.get(rel_type, 1.0)
            new_score = score * w / (depth + 1)
            if n not in visited or new_score > visited[n]:
                visited[n] = new_score
                queue.append((n, depth + 1, new_score))
                
    # Deduplicate keeping highest score
    unique_nodes = {}
    for node_id, data, score in retrieved:
        if node_id not in unique_nodes or score > unique_nodes[node_id][1]:
            unique_nodes[node_id] = (data, score)
            
    # Sort by score descending
    sorted_nodes = sorted(unique_nodes.items(), key=lambda x: x[1][1], reverse=True)
    
    selected_context = []
    selected_node_ids = []
    total_chars = 0
    
    # Context template matching LifecycleGuidedContextStrategy.construct_context
    for node_id, (data, score) in sorted_nodes:
        node_type = data.get("type", "unknown")
        stage = data.get("lifecycle_stage", "unknown")
        name = data.get("name", "unnamed")
        
        block_header = f"Node: {node_id} (Type: {node_type}, Stage: {stage})"
        if node_type == "commit":
            content = f"Message: {data.get('message', '')}\nAuthor: {data.get('author', '')}"
        elif node_type in ["issue", "pull_request"]:
            content = f"Title: {name}\nState: {data.get('state', '')}\nDescription: {data.get('text_content', '')}"
        else:
            content = f"Name: {name}\nPath: {data.get('path', '')}"
            
        node_text = f"{block_header}\n{content}\n"
        
        if total_chars + len(node_text) <= max_chars:
            selected_node_ids.append(node_id)
            selected_context.append(node_text)
            total_chars += len(node_text)
        else:
            break
            
    return selected_node_ids, total_chars

def evaluate_retrieval(retrieved_nodes, target_files):
    """Calculates Recall and MRR for files in the retrieved context."""
    # Filter retrieved nodes to those that are files
    retrieved_files = []
    for node_id in retrieved_nodes:
        # A file node ID contains 'source_code/' or 'tests/' etc.
        if "source_code/" in node_id or "tests/" in node_id or "documentation/" in node_id or "build_configs/" in node_id:
            retrieved_files.append(clean_file_path(node_id))
            
    # Remove duplicates while preserving order
    seen = set()
    unique_retrieved_files = [x for x in retrieved_files if not (x in seen or seen.add(x))]
    
    # Recall
    matched_targets = [f for f in unique_retrieved_files if f in target_files]
    recall = len(matched_targets) / len(target_files) if len(target_files) > 0 else 0.0
    
    # MRR (Reciprocal Rank of the first target file)
    mrr = 0.0
    for idx, f in enumerate(unique_retrieved_files):
        if f in target_files:
            mrr = 1.0 / (idx + 1)
            break
            
    return recall, mrr

def main():
    base_dir = "/Users/naveenmanohar/capstone_experiments"
    dataset_dir = "/Users/naveenmanohar/capstone_preprocessing/output/prepared_dataset"
    db_path = f"sqlite:///{os.path.join(base_dir, 'results', 'optuna_study.db')}"
    
    # 1. Load evaluation tasks
    loader = EvaluationLoader(os.path.join(base_dir, "data", "evaluation_tasks.json"))
    tasks = loader.load_tasks()
    logger.info(f"Loaded {len(tasks)} tasks for optimization.")
    
    # Target versions
    version_tags = sorted(list(set([t["target_version"] for t in tasks])))
    
    # Pre-load embedding indexers and lifecycle graphs to speed up tuning
    indexers = {}
    graphs = {}
    candidate_model = "sentence-transformers/all-mpnet-base-v2"
    cache_dir = os.path.join(base_dir, "cache", "embeddings", "optuna_all_mpnet")
    
    logger.info("Pre-building indexers and graphs for all target versions (this might take a minute)...")
    for tag in version_tags:
        # Indexer
        idx = rag_indexer.RAGIndexer(
            dataset_dir=dataset_dir,
            embedding_model_name=candidate_model,
            cache_dir=cache_dir
        )
        idx.build_index_for_version(tag)
        indexers[tag] = idx
        
        # Graph
        graph = LifecycleProjectGraph(dataset_dir)
        graph.build_graph_for_version(tag)
        graphs[tag] = graph
        
    logger.info("Pre-loading finished. Commencing hyperparameter optimization...")
    
    def objective(trial):
        # Sample hyperparameters
        top_k = trial.suggest_int("top_k", 2, 12)
        similarity_threshold = trial.suggest_float("similarity_threshold", 0.0, 0.7)
        max_depth = trial.suggest_int("max_depth", 1, 3)
        weight_tests_class = trial.suggest_float("weight_tests_class", 0.1, 4.0)
        weight_resolves_issue = trial.suggest_float("weight_resolves_issue", 0.1, 4.0)
        weight_modified_file = trial.suggest_float("weight_modified_file", 0.1, 4.0)
        max_context_chars = trial.suggest_int("max_context_chars", 2000, 12000)
        
        weights = {
            "tests_class": weight_tests_class,
            "resolves_issue": weight_resolves_issue,
            "modified_file": weight_modified_file
        }
        
        task_scores = []
        
        for task in tasks:
            tag = task["target_version"]
            query = task["description"]
            targets = [clean_file_path(f) for f in task["target_files"]]
            
            idx = indexers[tag]
            graph = graphs[tag]
            
            # Step 1: Semantic search
            search_matches = idx.search(query, top_k=top_k)
            
            # Step 2: Similarity threshold filter
            entry_nodes = [
                m["artifact_id"] for m in search_matches 
                if m["similarity_score"] >= similarity_threshold
            ]
            
            # Step 3: Graph traversal and context budgeting
            retrieved_nodes, total_chars = traverse_with_weights(
                graph=graph.graph,
                entry_nodes=entry_nodes,
                max_depth=max_depth,
                weights=weights,
                max_chars=max_context_chars
            )
            
            # Step 4: Metric calculations
            recall, mrr = evaluate_retrieval(retrieved_nodes, targets)
            
            # Context Efficiency Penalty (relative to sampled limit)
            efficiency = 1.0 - (total_chars / 12000.0)
            
            # Composite Equation
            score = (0.4 * recall) + (0.4 * mrr) + (0.2 * efficiency)
            task_scores.append(score)
            
        return float(np.mean(task_scores))
        
    # Setup Optuna Study
    study = optuna.create_study(
        study_name="lifecycle_hyperparameter_optimization",
        storage=db_path,
        direction="maximize",
        load_if_exists=True
    )
    
    # Run optimization for 80 trials to search comprehensively
    study.optimize(objective, n_trials=80)
    
    best_trial = study.best_trial
    logger.info(f"Optimization completed. Best Score: {best_trial.value:.4f}")
    
    # Structure optimized config (merging defaults to preserve all run options)
    config_in = os.path.join(base_dir, "config", "default_config.json")
    if os.path.exists(config_in):
        with open(config_in, "r") as f:
            full_config = json.load(f)
    else:
        full_config = {}
        
    optimized_params = {
        "embedding_model_name": candidate_model,
        "top_k": int(best_trial.params["top_k"]),
        "similarity_threshold": float(best_trial.params["similarity_threshold"]),
        "max_depth": int(best_trial.params["max_depth"]),
        "weight_tests_class": float(best_trial.params["weight_tests_class"]),
        "weight_resolves_issue": float(best_trial.params["weight_resolves_issue"]),
        "weight_modified_file": float(best_trial.params["weight_modified_file"]),
        "max_context_chars": int(best_trial.params["max_context_chars"])
    }
    
    full_config.update(optimized_params)
    
    # Save parameters to config/optimized_config.json
    config_out = os.path.join(base_dir, "config", "optimized_config.json")
    with open(config_out, "w") as f:
        json.dump(full_config, f, indent=2)
        
    logger.info(f"Best hyperparameters merged and exported successfully to {config_out}")

    
    # Format and print best parameters
    print("\n### Optuna Hyperparameter Optimization Completed")
    print(f"**Best Composite Retrieval Score**: `{best_trial.value:.4f}`\n")
    print("| Hyperparameter | Optimized Value | Description |")
    print("| :--- | :---: | :--- |")
    print(f"| **`top_k`** | `{optimized_params['top_k']}` | Quantity of initial semantic search entry points |")
    print(f"| **`similarity_threshold`** | `{optimized_params['similarity_threshold']:.4f}` | Minimum cosine similarity required for seed nodes |")
    print(f"| **`max_depth`** | `{optimized_params['max_depth']}` | Graph traversal BFS search depth |")
    print(f"| **`weight_tests_class`** | `{optimized_params['weight_tests_class']:.4f}` | Graph weight multiplier for tests-class relationships |")
    print(f"| **`weight_resolves_issue`** | `{optimized_params['weight_resolves_issue']:.4f}` | Graph weight multiplier for resolves-issue relationships |")
    print(f"| **`weight_modified_file`** | `{optimized_params['weight_modified_file']:.4f}` | Graph weight multiplier for modified-file relationships |")
    print(f"| **`max_context_chars`** | `{optimized_params['max_context_chars']} chars` | Maximum budget size of generated context block |")
    print()

if __name__ == "__main__":
    main()
