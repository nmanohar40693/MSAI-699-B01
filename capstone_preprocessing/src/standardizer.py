import logging

logger = logging.getLogger(__name__)

class MetadataStandardizer:
    def __init__(self):
        pass

    def standardize_file_metadata(self, file_artifacts: list) -> list:
        """Ensures file artifacts contain all standard metadata fields, filling defaults if missing."""
        standardized = []
        for art in file_artifacts:
            std_art = {
                "artifact_type": art.get("artifact_type", "unknown"),
                "artifact_id": art.get("artifact_id", ""),
                "name": art.get("name", ""),
                "path": art.get("path", ""),
                "extension": art.get("extension", ""),
                "size_bytes": art.get("size_bytes", 0),
                "lines_of_code": art.get("lines_of_code", 0),
                "raw_content_sha256": art.get("raw_content_sha256", ""),
                "last_modified_commit": art.get("last_modified_commit", ""),
                "associated_versions": art.get("associated_versions", []),
                "raw_content": art.get("raw_content", "")
            }
            
            # Carry over Java-specific metadata
            if "package" in art:
                std_art["package"] = art["package"]
            if "imports" in art:
                std_art["imports"] = art["imports"]
                
            standardized.append(std_art)
        return standardized

    def standardize_commit_metadata(self, commits: list) -> list:
        """Standardizes commit structures."""
        standardized = []
        for commit in commits:
            std_commit = {
                "artifact_type": "commit",
                "artifact_id": commit.get("hash", ""),
                "hash": commit.get("hash", ""),
                "author": commit.get("author", ""),
                "timestamp": commit.get("timestamp", ""),
                "message": commit.get("message", ""),
                "parents": commit.get("parents", []),
                "files_changed": commit.get("files_changed", []),
                "referenced_issues": commit.get("referenced_issues", []),
                "associated_versions": commit.get("associated_versions", [])
            }
            standardized.append(std_commit)
        return standardized

    def standardize_discussion_metadata(self, items: list, item_type: str) -> list:
        """Standardizes issues and pull requests."""
        standardized = []
        for item in items:
            std_item = {
                "artifact_type": item_type,
                "artifact_id": item.get("artifact_id", f"{item_type}-{item.get('number', 0)}"),
                "number": item.get("number", 0),
                "title": item.get("title", ""),
                "body": item.get("body", ""),
                "author": item.get("author", ""),
                "state": item.get("state", ""),
                "created_at": item.get("created_at", ""),
                "closed_at": item.get("closed_at", ""),
                "comments": [
                    {
                        "author": c.get("author", ""),
                        "body": c.get("body", ""),
                        "timestamp": c.get("timestamp", "")
                    }
                    for c in item.get("comments", [])
                ],
                "linked_commits": item.get("linked_commits", []),
                "associated_versions": item.get("associated_versions", [])
            }
            standardized.append(std_item)
        return standardized
