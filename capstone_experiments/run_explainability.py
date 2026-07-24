import os
import json
import logging
import resource
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import shap

import src.rag_indexer as rag_indexer
from src.graph_builder import LifecycleProjectGraph
from src.evaluation import EvaluationLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_explainability")

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

def get_implied_stage(query: str) -> str:
    """Infers the target lifecycle stage from task query keywords."""
    query_l = query.lower()
    if "test" in query_l:
        return "Testing"
    elif any(k in query_l for k in ["fix", "bug", "issue", "error", "defect"]):
        return "Debugging"
    elif any(k in query_l for k in ["implement", "add", "controller", "create", "model", "service"]):
        return "Implementation"
    return "Maintenance"

def get_implied_artifact_type(query: str) -> str:
    """Infers the target artifact type from task query keywords."""
    query_l = query.lower()
    if "test" in query_l:
        return "test_case"
    elif any(k in query_l for k in ["commit", "hash", "revision"]):
        return "commit"
    elif any(k in query_l for k in ["issue", "ticket", "bug"]):
        return "issue"
    return "source_code"

def build_features_for_task(task, indexers, graphs, model):
    """Computes features for all candidate nodes related to a single task."""
    tag = task["target_version"]
    query = task["description"]
    targets = [clean_file_path(f) for f in task["target_files"]]
    
    idx_instance = indexers[tag]
    graph_instance = graphs[tag].graph
    
    # 1. Get seed nodes (entry points via RAG search)
    search_matches = idx_instance.search(query, top_k=6)
    seed_nodes = [m["artifact_id"] for m in search_matches]
    
    # 2. Compile candidate nodes (target files + BFS traversed neighbors up to depth 3)
    candidate_set = set(task["target_files"]) # guaranteed inclusion
    
    # Run a simple BFS traversal from seed nodes up to depth 3 to collect candidate context
    visited = set(seed_nodes)
    queue = [(node, 0) for node in seed_nodes if graph_instance.has_node(node)]
    
    while queue:
        curr, depth = queue.pop(0)
        candidate_set.add(curr)
        if depth < 3:
            neighbors = list(graph_instance.successors(curr)) + list(graph_instance.predecessors(curr))
            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, depth + 1))
                    
    # Filter candidate set to nodes that actually exist in the graph
    candidates = [node for node in candidate_set if graph_instance.has_node(node)]
    
    # Encode query once
    query_vector = model.encode([query])[0]
    
    # Recency mapping
    recency_map = {"b11afec3": 0.33, "b3ee2c53": 0.66, "51045d16": 1.0}
    recency = recency_map.get(tag, 1.0)
    
    # Implied task targets
    implied_stage = get_implied_stage(query)
    implied_type = get_implied_artifact_type(query)
    
    # Max degree for normalization
    degrees = dict(graph_instance.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    task_rows = []
    
    for node in candidates:
        data = graph_instance.nodes[node]
        node_type = data.get("type", "unknown")
        node_text = data.get("text_content", "")
        node_name = data.get("name", "")
        
        # 1. Semantic Similarity
        # Encode on the fly if it is a non-file node, else fetch from RAG index if cached
        similarity = 0.0
        if node_text:
            node_vector = model.encode([node_text])[0]
            similarity = float(np.dot(query_vector, node_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(node_vector) + 1e-10))
            
        # 2. Symbol Match
        symbol_match = 0.0
        clean_name = node_name.lower().replace(".java", "").replace("tests", "").replace("test", "")
        if clean_name and clean_name in query.lower():
            symbol_match = 1.0
            
        # 3. Graph Distance (Shortest path distance from any seed node)
        min_dist = 99.0
        # Convert graph to undirected for shortest path calculation
        undirected_g = graph_instance.to_undirected()
        for seed in seed_nodes:
            if undirected_g.has_node(seed) and undirected_g.has_node(node):
                try:
                    d = nx.shortest_path_length(undirected_g, source=seed, target=node)
                    if d < min_dist:
                        min_dist = float(d)
                except nx.NetworkXNoPath:
                    pass
        # Decay normalizer (1.0 = direct match, 0.0 = unreachable)
        graph_dist_score = 1.0 / (min_dist + 1)
        
        # 4. Traceability Degree Centrality
        degree_score = degrees.get(node, 0) / max_degree
        
        # 5. Test Source Link
        test_source_link = 0.0
        for u, v, k in graph_instance.edges(node, data=True):
            if k.get("relationship_type") == "tests_class":
                test_source_link = 1.0
                break
                
        # 6. Lifecycle Stage Match
        stage_match = 1.0 if data.get("lifecycle_stage") == implied_stage else 0.0
        
        # 7. Artifact Type Match
        type_match = 1.0 if node_type == implied_type else 0.0
        
        # Binary Label indicating relevance
        label = 1 if clean_file_path(node) in targets else 0
        
        task_rows.append({
            "task_id": task["task_id"],
            "node_id": node,
            "semantic_similarity": similarity,
            "symbol_match": symbol_match,
            "graph_distance": graph_dist_score,
            "traceability_degree": degree_score,
            "test_source_link": test_source_link,
            "lifecycle_stage_match": stage_match,
            "artifact_type_match": type_match,
            "version_recency": recency,
            "label": label
        })
        
    return task_rows

def main():
    base_dir = "/Users/naveenmanohar/capstone_experiments"
    dataset_dir = "/Users/naveenmanohar/capstone_preprocessing/output/prepared_dataset"
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load 10 tasks
    loader = EvaluationLoader(os.path.join(base_dir, "data", "evaluation_tasks.json"))
    tasks = loader.load_tasks()
    
    # 2. Pre-build indexers and graphs
    version_tags = sorted(list(set([t["target_version"] for t in tasks])))
    indexers = {}
    graphs = {}
    candidate_model = "sentence-transformers/all-mpnet-base-v2"
    cache_dir = os.path.join(base_dir, "cache", "embeddings", "optuna_all_mpnet")
    
    # Load model
    logger.info(f"Loading SentenceTransformer model: {candidate_model}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(candidate_model)
    
    logger.info("Building indices and graphs...")
    for tag in version_tags:
        idx = rag_indexer.RAGIndexer(dataset_dir=dataset_dir, embedding_model_name=candidate_model, cache_dir=cache_dir)
        idx.build_index_for_version(tag)
        indexers[tag] = idx
        
        graph = LifecycleProjectGraph(dataset_dir)
        graph.build_graph_for_version(tag)
        graphs[tag] = graph
        
    # 3. Build Feature Matrix
    logger.info("Extracting features for all candidate nodes...")
    dataset = []
    for task in tasks:
        rows = build_features_for_task(task, indexers, graphs, model)
        dataset.extend(rows)
        
    df = pd.DataFrame(dataset)
    df.to_csv(os.path.join(results_dir, "feature_matrix.csv"), index=False)
    logger.info(f"Feature matrix compiled. Shape: {df.shape}. Saved to results/feature_matrix.csv")
    
    # 4. Train surrogate classifier model
    feature_cols = [
        "semantic_similarity", "symbol_match", "graph_distance", 
        "traceability_degree", "test_source_link", "lifecycle_stage_match", 
        "artifact_type_match", "version_recency"
    ]
    
    X = df[feature_cols]
    y = df["label"]
    
    # Check class distribution
    class_dist = y.value_counts().to_dict()
    logger.info(f"Class distribution in feature matrix: {class_dist}")
    
    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X, y)
    
    # Evaluate
    preds = rf.predict(X)
    probs = rf.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    logger.info(f"Surrogate model training complete. Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y, preds)}")
    
    # 5. Compute SHAP Values
    logger.info("Calculating TreeSHAP contributions...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    
    # Handle SHAP output format
    if isinstance(shap_values, list):
        sv = shap_values[1] # select class 1 (relevance)
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1] # shape (N, M, 2)
    else:
        sv = shap_values
        
    # 6. Plot Global summary
    logger.info("Generating global SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X, plot_type="bar", show=False)
    plt.title("TreeSHAP Global Feature Importance Summary", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "shap_global_summary.png"), bbox_inches="tight")
    plt.close()
    
    # 7. Plot Local Representative Cases
    # Add predictions and probs to df for indexing
    df["prediction"] = preds
    df["probability"] = probs
    
    cases = {
        "true_positive": df[(df["label"] == 1) & (df["prediction"] == 1)],
        "false_positive": df[(df["label"] == 0) & (df["prediction"] == 1)],
        "false_negative": df[(df["label"] == 1) & (df["prediction"] == 0)],
    }
    
    # Borderline: closest probability to 0.5 (regardless of prediction)
    df["dist_from_borderline"] = (df["probability"] - 0.5).abs()
    borderline_df = df.sort_values("dist_from_borderline")
    cases["borderline"] = borderline_df.head(1) if not borderline_df.empty else pd.DataFrame()
    
    logger.info("Generating local surrogate force bar plots...")
    
    for case_name, case_df in cases.items():
        if case_df.empty:
            logger.warning(f"No instances found for representative case: {case_name}. Plot skipped.")
            continue
            
        # Select first instance
        inst_idx = case_df.index[0]
        node_id = df.loc[inst_idx, "node_id"]
        task_id = df.loc[inst_idx, "task_id"]
        prob = df.loc[inst_idx, "probability"]
        
        # Build explanation object
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        explanation = shap.Explanation(
            values=sv[inst_idx],
            base_values=base_val,
            data=X.iloc[inst_idx].values,
            feature_names=feature_cols
        )
        
        plt.figure(figsize=(8, 5))
        shap.plots.bar(explanation, show=False)
        plt.title(f"SHAP Local Explanation ({case_name.upper().replace('_', ' ')})\nNode: {node_id} (Task: {task_id}, Prob: {prob:.4f})", fontsize=12, pad=15)
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f"shap_local_{case_name}.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved local plot: {plot_path}")
        
    print("\n### Feature Engineering and Surrogate Explainability Analysis Completed")
    print(f"Features mapped: `{len(feature_cols)}`")
    print(f"Dataset size: `{len(df)} rows`")
    print(f"Surrogate model accuracy: `{acc:.4%}`\n")

if __name__ == "__main__":
    main()
