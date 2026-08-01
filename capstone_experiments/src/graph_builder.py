import os
import json
import logging
import networkx as nx

logger = logging.getLogger(__name__)

class LifecycleProjectGraph:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.graph = nx.DiGraph()

    def build_graph_for_version(self, version_tag: str):
        """Constructs the Lifecycle-Guided Project Knowledge Graph for the specified version."""
        self.graph.clear()
        
        # Paths
        version_dir = os.path.join(self.dataset_dir, "versions", version_tag)
        if not os.path.exists(version_dir):
            raise FileNotFoundError(f"Version directory not found: {version_dir}")
            
        # 1. Load Preprocessed Datasets
        commits = self._load_json_file("commits.json", [])
        issues = self._load_json_file("issues.json", [])
        prs = self._load_json_file("pull_requests.json", [])
        relationships = self._load_json_file("relationships.json", [])
        
        # Filter files for the target version
        files_meta = self._load_version_files_meta(version_dir)
        
        # 2. Add Nodes with Metadata and Lifecycle Stages
        # Classify nodes into lifecycle stages:
        # Requirements, Design, Implementation, Debugging, Testing, Code review, Documentation, Maintenance
        
        # Files Nodes (Implementation, Testing, Documentation, Maintenance)
        for fm in files_meta:
            node_id = fm["artifact_id"]
            category = fm["artifact_type"]
            
            # Map category to lifecycle stage
            stage = "Implementation"
            if category == "test_case":
                stage = "Testing"
            elif category == "documentation":
                stage = "Documentation"
            elif category == "build_config":
                stage = "Maintenance"
                
            self.graph.add_node(
                node_id,
                artifact_id=node_id,
                name=fm["name"],
                type=category,
                lifecycle_stage=stage,
                path=fm["path"],
                version=version_tag,
                text_content=fm.get("package", "") + "\n" + ", ".join(fm.get("imports", []))
            )

        # Commits Nodes (Maintenance/Implementation)
        version_commits = [c for c in commits if version_tag in c.get("associated_versions", [])]
        for c in version_commits:
            node_id = c["hash"]
            self.graph.add_node(
                node_id,
                artifact_id=node_id,
                name=f"Commit: {node_id[:8]}",
                type="commit",
                lifecycle_stage="Implementation",
                message=c["message"],
                author=c["author"],
                timestamp=c["timestamp"],
                text_content=c["message"]
            )

        # Issues Nodes (Requirements/Debugging)
        version_issues = [i for i in issues if version_tag in i.get("associated_versions", [])]
        for issue in version_issues:
            node_id = issue["artifact_id"]
            # Map state to stage (open = Requirement, closed = Debugging)
            stage = "Debugging" if issue["state"] == "closed" else "Requirements"
            self.graph.add_node(
                node_id,
                artifact_id=node_id,
                name=issue["title"],
                type="issue",
                lifecycle_stage=stage,
                number=issue["number"],
                state=issue["state"],
                text_content=f"{issue['title']}\n{issue['body']}"
            )

        # Pull Request Nodes (Code Review/Debugging)
        version_prs = [p for p in prs if version_tag in p.get("associated_versions", [])]
        for pr in version_prs:
            node_id = pr["artifact_id"]
            self.graph.add_node(
                node_id,
                artifact_id=node_id,
                name=pr["title"],
                type="pull_request",
                lifecycle_stage="Code review",
                number=pr["number"],
                state=pr["state"],
                text_content=f"{pr['title']}\n{pr['body']}"
            )

        # 3. Add Edges representing traceability relationships
        # Standardize relationships from relationships.json
        for rel in relationships:
            src = rel["source_id"]
            tgt = rel["target_id"]
            
            # Map commit hashes and file paths to match graph node IDs
            resolved_src = self._resolve_node_id(src, version_tag, files_meta)
            resolved_tgt = self._resolve_node_id(tgt, version_tag, files_meta)
            
            if resolved_src and resolved_tgt:
                # Add edge if both nodes are present in our graph for this version
                if self.graph.has_node(resolved_src) and self.graph.has_node(resolved_tgt):
                    self.graph.add_edge(
                        resolved_src,
                        resolved_tgt,
                        relationship_type=rel["relationship_type"],
                        weight=rel.get("weight", 1.0),
                        is_explicit=True,
                        confidence_score=1.0
                    )

        logger.info(f"Built Lifecycle-Guided Knowledge Graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def get_graph_statistics(self) -> dict:
        """Returns structural statistics of the built graph."""
        if not self.graph:
            return {}
            
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1
            
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            t = data.get("relationship_type", "unknown")
            edge_types[t] = edge_types.get(t, 0) + 1
            
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            "connected_components": nx.number_weakly_connected_components(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }

    def traverse_for_context(self, entry_nodes: list, max_depth: int = 2) -> list:
        """Traverses the graph starting from entry points to gather historically connected lifecycle artifacts."""
        visited = set()
        retrieved_nodes = []
        
        # Queue stores tuples of (node_id, current_depth)
        queue = [(node, 0) for node in entry_nodes if self.graph.has_node(node)]
        
        for node, _ in queue:
            visited.add(node)
            
        while queue:
            curr_node, depth = queue.pop(0)
            
            # Fetch node metadata
            node_data = self.graph.nodes[curr_node]
            retrieved_nodes.append((curr_node, node_data, depth))
            
            if depth >= max_depth:
                continue
                
            # Traverse outgoing and incoming edges to gather complete context
            neighbors = list(self.graph.successors(curr_node)) + list(self.graph.predecessors(curr_node))
            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, depth + 1))
                    
        return retrieved_nodes

    def traverse_with_weights(self, entry_nodes: list, max_depth: int, weights: dict, max_chars: int) -> list:
        """BFS graph traversal with edge weights, score ranking, and context character budgeting."""
        # queue elements: (node_id, depth, score)
        queue = [(node, 0, 1.0) for node in entry_nodes if self.graph.has_node(node)]
        visited = {}
        for node, _, score in queue:
            visited[node] = (score, 0)
            
        retrieved = []
        
        while queue:
            curr, depth, score = queue.pop(0)
            retrieved.append((curr, self.graph.nodes[curr], depth, score))
            
            if depth >= max_depth:
                continue
                
            # Successors
            for n in self.graph.successors(curr):
                rel_type = self.graph[curr][n].get('relationship_type', 'unknown')
                w = weights.get(rel_type, 1.0)
                new_score = score * w / (depth + 1)
                if n not in visited or new_score > visited[n][0]:
                    visited[n] = (new_score, depth + 1)
                    queue.append((n, depth + 1, new_score))
                    
            # Predecessors
            for n in self.graph.predecessors(curr):
                rel_type = self.graph[n][curr].get('relationship_type', 'unknown')
                w = weights.get(rel_type, 1.0)
                new_score = score * w / (depth + 1)
                if n not in visited or new_score > visited[n][0]:
                    visited[n] = (new_score, depth + 1)
                    queue.append((n, depth + 1, new_score))
                    
        self.last_visited_count = len(visited)
        self.last_max_depth_visited = max([depth for _, depth in visited.values()]) if visited else 0

        # Deduplicate keeping highest score
        unique_nodes = {}
        for node_id, data, depth, score in retrieved:
            if node_id not in unique_nodes or score > unique_nodes[node_id][2]:
                unique_nodes[node_id] = (data, depth, score)
                
        # Sort by score descending
        sorted_nodes = sorted(unique_nodes.items(), key=lambda x: x[1][2], reverse=True)
        
        selected_nodes = []
        total_chars = 0
        
        # Format character budget check
        for node_id, (data, depth, score) in sorted_nodes:
            node_type = data.get("type", "unknown")
            stage = data.get("lifecycle_stage", "unknown")
            name = data.get("name", "unnamed")
            
            block_header = f"Node: {node_id} (Type: {node_type}, Stage: {stage}, Relation Depth: {depth})"
            
            if node_type == "commit":
                content = f"Message: {data.get('message', '')}\nAuthor: {data.get('author', '')}\nTimestamp: {data.get('timestamp', '')}"
            elif node_type in ["issue", "pull_request"]:
                content = f"Title: {name}\nState: {data.get('state', '')}\nDescription: {data.get('text_content', '')}"
            else:
                content = f"Name: {name}\nPath: {data.get('path', '')}"
                
            node_text = f"{block_header}\n{content}\n"
            
            if total_chars + len(node_text) <= max_chars:
                selected_nodes.append((node_id, data, depth))
                total_chars += len(node_text)
            else:
                break
                
        return selected_nodes


    def _resolve_node_id(self, raw_id: str, version_tag: str, files_meta: list) -> str:
        """Resolves file paths or hashes to the exact node identifier inside the graph."""
        # Check if raw_id is a file path and match with artifact_id
        for fm in files_meta:
            if fm["path"] == raw_id or fm["artifact_id"] == raw_id:
                return fm["artifact_id"]
        # Commits/Issues/PRs are resolved directly
        return raw_id

    def _load_json_file(self, filename: str, default):
        path = os.path.join(self.dataset_dir, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return default

    def _load_version_files_meta(self, version_dir: str) -> list:
        meta_file = os.path.join(version_dir, "version_meta.json")
        if not os.path.exists(meta_file):
            return []
            
        # Files metadata list is inside the version_meta.json under snapshots mapping or version snapshot output
        # Let's inspect files in directory recursively if meta is simple
        # Actually, in aligner.py, we wrote out version snapshot paths in the list returned by _write_snapshot_files
        # Let's read files from the folders recursively to compile the current files metadata
        files_meta = []
        subdirs = ["source_code", "tests", "documentation", "build_configs"]
        
        for sub in subdirs:
            sub_path = os.path.join(version_dir, sub)
            if not os.path.exists(sub_path):
                continue
            for root, _, files in os.walk(sub_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, version_dir)
                    
                    category = "source_code"
                    if sub == "tests":
                        category = "test_case"
                    elif sub == "documentation":
                        category = "documentation"
                    elif sub == "build_configs":
                        category = "build_config"
                        
                    files_meta.append({
                        "artifact_id": f"versions/{os.path.basename(version_dir)}/{rel_path}",
                        "artifact_type": category,
                        "name": file,
                        "path": rel_path.replace(f"{sub}/", "") # standard file path
                    })
        return files_meta
