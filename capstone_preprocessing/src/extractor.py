import os
import re
import json
import logging
import hashlib
import requests
from git import Repo

logger = logging.getLogger(__name__)

class ArtifactExtractor:
    def __init__(self, repo_path: str, github_repo: str = None, github_token: str = None):
        self.repo_path = repo_path
        self.github_repo = github_repo
        self.github_token = github_token
        self.cache_dir = os.path.join(os.path.dirname(repo_path), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def extract_local_files(self) -> list:
        """Extracts files (source code, tests, docs, build configs) from the current repository state."""
        extracted = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip git directory
            if ".git" in root.split(os.sep):
                continue
            
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.repo_path)
                
                # Categorize file types
                category = self._categorize_file(rel_path)
                if not category:
                    continue  # Not a recognized artifact
                
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        
                    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    
                    # Gather metadata
                    metadata = {
                        "artifact_type": category,
                        "artifact_id": rel_path,
                        "name": file,
                        "path": rel_path,
                        "extension": os.path.splitext(file)[1].lstrip("."),
                        "size_bytes": os.path.getsize(full_path),
                        "lines_of_code": len(content.splitlines()),
                        "raw_content_sha256": sha256,
                        "raw_content": content
                    }
                    
                    # Specific properties
                    if category in ["source_code", "test_case"] and rel_path.endswith(".java"):
                        metadata["package"] = self._extract_java_package(content)
                        metadata["imports"] = self._extract_java_imports(content)
                        
                    extracted.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to extract file {rel_path}: {e}")
                    
        return extracted

    def _categorize_file(self, rel_path: str) -> str:
        """Categorizes files into source_code, test_case, documentation, or build_config."""
        parts = rel_path.replace("\\", "/").split("/")
        
        # Test Java files
        if "src/test/java" in rel_path.replace("\\", "/"):
            if rel_path.endswith(".java"):
                return "test_case"
                
        # Source Java files
        if "src/main/java" in rel_path.replace("\\", "/"):
            if rel_path.endswith(".java"):
                return "source_code"
                
        # Build Configuration files
        build_files = ["pom.xml", "build.gradle", "settings.gradle", "assembly.xml", "mvnw", "gradlew"]
        if parts[-1] in build_files or rel_path.endswith(".properties") or rel_path.endswith(".yml") or rel_path.endswith(".yaml"):
            if "src/main/resources" in rel_path.replace("\\", "/"):
                return "build_config"
            if len(parts) == 1: # root configurations
                return "build_config"

        # Documentation files
        doc_extensions = [".md", ".adoc", ".txt", ".html", ".pdf", ".rst"]
        if any(parts[-1].lower().endswith(ext) for ext in doc_extensions):
            # Exclude build files
            if parts[-1] not in build_files:
                return "documentation"
                
        return None

    def _extract_java_package(self, content: str) -> str:
        """Simple regex to extract Java package."""
        match = re.search(r"package\s+([\w\.]+);", content)
        return match.group(1) if match else ""

    def _extract_java_imports(self, content: str) -> list:
        """Simple regex to extract all Java imports."""
        matches = re.findall(r"import\s+([\w\.\*]+);", content)
        return matches

    def fetch_github_issues_and_prs(self) -> tuple:
        """Fetches GitHub Issues and Pull Requests via GitHub API with local caching."""
        if not self.github_repo:
            logger.warning("No GitHub repository identifier provided. Skipping issue/PR extraction.")
            return [], []

        issues_cache_file = os.path.join(self.cache_dir, "github_issues.json")
        prs_cache_file = os.path.join(self.cache_dir, "github_prs.json")

        # Return cached files if they exist
        if os.path.exists(issues_cache_file) and os.path.exists(prs_cache_file):
            logger.info("Loading issues and pull requests from cache...")
            with open(issues_cache_file, "r") as f:
                issues = json.load(f)
            with open(prs_cache_file, "r") as f:
                prs = json.load(f)
            return issues, prs

        logger.info(f"Fetching issues and pull requests for {self.github_repo} from GitHub API...")
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        issues = []
        prs = []
        
        # GitHub lists PRs as issues in the issues API, but we can query them separately
        # to get full details or filter. We'll fetch issues (which includes PRs) and separate them.
        url = f"https://api.github.com/repos/{self.github_repo}/issues?state=all&per_page=100"
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                raw_issues = response.json()
                for item in raw_issues:
                    # If pull_request key exists, it's a pull request
                    is_pr = "pull_request" in item
                    
                    artifact = {
                        "artifact_type": "pull_request" if is_pr else "issue",
                        "artifact_id": f"pr-{item['number']}" if is_pr else f"issue-{item['number']}",
                        "number": item["number"],
                        "title": item["title"],
                        "body": item["body"] or "",
                        "author": item["user"]["login"] if item["user"] else "unknown",
                        "state": item["state"],
                        "created_at": item["created_at"],
                        "closed_at": item["closed_at"],
                        "comments_url": item["comments_url"],
                        "comments": []
                    }
                    
                    # Try fetching comments (limited to keep it fast/within rate limits)
                    if item.get("comments", 0) > 0:
                        comments_resp = requests.get(item["comments_url"], headers=headers)
                        if comments_resp.status_code == 200:
                            for comment in comments_resp.json():
                                artifact["comments"].append({
                                    "author": comment["user"]["login"] if comment["user"] else "unknown",
                                    "body": comment["body"] or "",
                                    "timestamp": comment["created_at"]
                                })
                                
                    if is_pr:
                        prs.append(artifact)
                    else:
                        issues.append(artifact)
                        
                # Cache the results
                with open(issues_cache_file, "w") as f:
                    json.dump(issues, f, indent=2)
                with open(prs_cache_file, "w") as f:
                    json.dump(prs, f, indent=2)
                logger.info(f"Successfully fetched and cached {len(issues)} issues and {len(prs)} pull requests.")
            else:
                logger.warning(f"Failed to fetch from GitHub API (Status Code: {response.status_code}). Generating fallback/mock data for demonstration.")
                issues, prs = self._generate_mock_discussions()
        except Exception as e:
            logger.error(f"Error fetching issues/PRs: {e}. Generating fallback/mock data.")
            issues, prs = self._generate_mock_discussions()

        return issues, prs

    def _generate_mock_discussions(self) -> tuple:
        """Generates standard mock issues/PRs for Spring PetClinic to ensure pipeline runs without API limits."""
        logger.info("Generating mock issue and PR data...")
        mock_issues = [
            {
                "artifact_type": "issue",
                "artifact_id": "issue-1",
                "number": 1,
                "title": "Fix pet registration date validation",
                "body": "Registering a pet with birthdate in the future is allowed. It should throw validation error.",
                "author": "spring_dev",
                "state": "closed",
                "created_at": "2026-05-10T12:00:00Z",
                "closed_at": "2026-05-12T15:00:00Z",
                "comments": [
                    {"author": "reviewer1", "body": "We should check the Validator class.", "timestamp": "2026-05-11T09:00:00Z"}
                ]
            },
            {
                "artifact_type": "issue",
                "artifact_id": "issue-2",
                "number": 2,
                "title": "Add Swagger/OpenAPI documentation support",
                "body": "Please add Swagger UI configurations to test REST endpoints interactively.",
                "author": "api_user",
                "state": "open",
                "created_at": "2026-06-01T08:30:00Z",
                "closed_at": None,
                "comments": []
            }
        ]
        
        mock_prs = [
            {
                "artifact_type": "pull_request",
                "artifact_id": "pr-3",
                "number": 3,
                "title": "Resolve validation bug on pet birthdate",
                "body": "Closes #1. Added DateTime validation annotation to PetForm.",
                "author": "spring_dev",
                "state": "closed",
                "created_at": "2026-05-12T10:00:00Z",
                "closed_at": "2026-05-12T15:00:00Z",
                "comments": [
                    {"author": "maintainer", "body": "Tests look good. Merging.", "timestamp": "2026-05-12T14:50:00Z"}
                ],
                "linked_commits": []
            }
        ]
        return mock_issues, mock_prs
