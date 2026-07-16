import re
import logging

logger = logging.getLogger(__name__)

class RelationshipPreserver:
    def __init__(self):
        self.relationships = []

    def identify_relationships(self, file_artifacts: dict, commits: list, issues: list, prs: list):
        """Identifies and stores traceability links across all software engineering artifacts.
        
        Args:
            file_artifacts: dict mapping versions to lists of file metadata.
            commits: list of standardized commit metadata.
            issues: list of standardized issues.
            prs: list of standardized pull requests.
        """
        self.relationships = []

        # 1. Commit to File modifications (using commit log change lists)
        self._map_commit_to_file_links(commits)

        # 2. Test Case to Source Code mapping (within each version snapshot)
        self._map_test_to_source_links(file_artifacts)

        # 3. Issue and PR linking (using regex for reference numbers in messages/bodies)
        self._map_discussion_links(commits, issues, prs)

        logger.info(f"Mapped {len(self.relationships)} total traceability relationships.")
        return self.relationships

    def _add_relationship(self, source_id: str, target_id: str, rel_type: str, weight: float = 1.0):
        """Helper to append a structured relationship tuple."""
        self.relationships.append({
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": rel_type,
            "weight": weight
        })

    def _map_commit_to_file_links(self, commits: list):
        """Maps commits to the source code/configuration files they modified."""
        for commit in commits:
            commit_id = commit["artifact_id"]
            for fc in commit.get("files_changed", []):
                file_path = fc.get("path")
                if file_path:
                    # Link commit to file
                    self._add_relationship(
                        source_id=commit_id,
                        target_id=file_path,
                        rel_type="modified_file",
                        weight=1.0
                    )

    def _map_test_to_source_links(self, file_artifacts_by_version: dict):
        """Heuristically links test classes to their corresponding implementation classes."""
        for version, artifacts in file_artifacts_by_version.items():
            # Separate tests and source files
            tests = [a for a in artifacts if a["artifact_type"] == "test_case"]
            sources = [a for a in artifacts if a["artifact_type"] == "source_code"]
            
            # Map source filenames without extensions for quick lookup
            source_map = {os.path.splitext(s["name"])[0]: s for s in sources}
            
            for test in tests:
                test_name = os.path.splitext(test["name"])[0]
                test_id = test["artifact_id"]
                
                # Match heuristic: "SomethingTest" or "SomethingTests" maps to "Something"
                matched_source_class = None
                for suffix in ["Test", "Tests", "TestCase"]:
                    if test_name.endswith(suffix):
                        candidate = test_name[:-len(suffix)]
                        if candidate in source_map:
                            matched_source_class = source_map[candidate]
                            break
                            
                if matched_source_class:
                    self._add_relationship(
                        source_id=test_id,
                        target_id=matched_source_class["artifact_id"],
                        rel_type="tests_class",
                        weight=1.0
                    )

    def _map_discussion_links(self, commits: list, issues: list, prs: list):
        """Links issues/PRs to commits and to each other using number references and hashes."""
        # Map of issue number -> issue_id
        issue_number_map = {i["number"]: i["artifact_id"] for i in issues}
        pr_number_map = {p["number"]: p["artifact_id"] for p in prs}

        # Regex to find issue reference numbers (e.g., "#12", "closes #104", "fixes #1")
        ref_regex = re.compile(r"#(\d+)")

        # Link commits to referenced issues/PRs
        for commit in commits:
            commit_id = commit["artifact_id"]
            message = commit.get("message", "")
            matches = ref_regex.findall(message)
            
            for m in matches:
                num = int(m)
                # Check if it references an issue
                if num in issue_number_map:
                    self._add_relationship(
                        source_id=commit_id,
                        target_id=issue_number_map[num],
                        rel_type="references_issue"
                    )
                # Check if it references a PR
                if num in pr_number_map:
                    self._add_relationship(
                        source_id=commit_id,
                        target_id=pr_number_map[num],
                        rel_type="part_of_pr"
                    )

        # Link PRs to issues and commits mentioned in their bodies or comments
        for pr in prs:
            pr_id = pr["artifact_id"]
            body = pr.get("body", "")
            
            # Extract references from body
            matches = ref_regex.findall(body)
            for comment in pr.get("comments", []):
                matches.extend(ref_regex.findall(comment.get("body", "")))
                
            for m in set(matches):
                num = int(m)
                if num in issue_number_map:
                    self._add_relationship(
                        source_id=pr_id,
                        target_id=issue_number_map[num],
                        rel_type="resolves_issue"
                    )
                    
            # Link commits explicitly named in PR body (e.g. SHA-1 hashes)
            sha_regex = re.compile(r"\b([a-f0-9]{7,40})\b")
            sha_matches = sha_regex.findall(body)
            for sha in sha_matches:
                # Link if this commit hash is in our commits
                # Find matching commit hexsha
                for commit in commits:
                    if commit["hash"].startswith(sha):
                        self._add_relationship(
                            source_id=pr_id,
                            target_id=commit["artifact_id"],
                            rel_type="includes_commit"
                        )
                        break
import os
