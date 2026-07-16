import os
import shutil
import logging
from datetime import datetime
from git import Repo

logger = logging.getLogger(__name__)

class VersionAligner:
    def __init__(self, repo_path: str, output_base_dir: str):
        self.repo_path = repo_path
        self.output_base_dir = output_base_dir
        self.repo = Repo(repo_path)

    def get_tag_info(self, tag_names: list) -> list:
        """Collects commit hashes and commit times for the specified tags, commits, or branches."""
        tag_info = []
        for name in tag_names:
            try:
                # Check if it's a tag in the repository
                if name in [t.name for t in self.repo.tags]:
                    tag = self.repo.tags[name]
                    commit = tag.commit
                    commit_time = commit.committed_datetime
                    tag_info.append({
                        "version": name,
                        "commit_hash": commit.hexsha,
                        "release_time": commit_time
                    })
                else:
                    # Treat it as a commit hash or branch name
                    commit = self.repo.commit(name)
                    commit_time = commit.committed_datetime
                    tag_info.append({
                        "version": name[:8] if len(name) > 12 else name,
                        "commit_hash": commit.hexsha,
                        "release_time": commit_time
                    })
            except Exception as e:
                logger.warning(f"Could not resolve version identifier '{name}': {e}. Skipping.")
                
        # Sort by release date to establish chronology
        tag_info.sort(key=lambda x: x["release_time"])
        return tag_info


    def extract_version_snapshots(self, version_info: list, extract_callback) -> dict:
        """Checks out each version sequentially, extracts files, and stores snapshots in output."""
        snapshots = {}
        original_active_branch = self.repo.active_branch.name if not self.repo.head.is_detached else "main"

        try:
            for i, info in enumerate(version_info):
                version = info["version"]
                commit_hash = info["commit_hash"]
                logger.info(f"Extracting snapshot for version {version} at commit {commit_hash[:8]}...")
                
                # Git checkout
                self.repo.git.checkout(commit_hash)
                
                # Extract local files using callback
                raw_files = extract_callback()
                
                # Create version directories
                version_dir = os.path.join(self.output_base_dir, "versions", version)
                os.makedirs(version_dir, exist_ok=True)
                
                # Organize files in version directories
                organized_files = self._write_snapshot_files(version_dir, raw_files)
                snapshots[version] = organized_files
                
                # Create version_meta.json
                prev_time = version_info[i-1]["release_time"].isoformat() if i > 0 else "1970-01-01T00:00:00Z"
                meta = {
                    "version": version,
                    "commit_hash": commit_hash,
                    "release_date": info["release_time"].isoformat(),
                    "time_boundary_start": prev_time,
                    "time_boundary_end": info["release_time"].isoformat(),
                    "files_count": len(raw_files)
                }
                with open(os.path.join(version_dir, "version_meta.json"), "w") as f:
                    import json
                    json.dump(meta, f, indent=2)

        finally:
            # Restore repository state
            logger.info(f"Restoring repository state to branch: {original_active_branch}")
            try:
                self.repo.git.checkout(original_active_branch)
            except Exception as e:
                logger.warning(f"Could not restore original branch: {e}. Checking out 'main'.")
                self.repo.git.checkout("main")
                
        return snapshots

    def _write_snapshot_files(self, version_dir: str, file_artifacts: list) -> list:
        """Copies raw content to designated subfolders in the version output directory and returns list of paths."""
        written = []
        for art in file_artifacts:
            category = art["artifact_type"]
            rel_path = art["path"]
            
            # Map category to folder name
            folder_map = {
                "source_code": "source_code",
                "test_case": "tests",
                "documentation": "documentation",
                "build_config": "build_configs"
            }
            folder_name = folder_map.get(category, "misc")
            
            dest_path = os.path.join(version_dir, folder_name, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(art["raw_content"])
                
            # Keep path relative to version_dir for portable dataset
            written.append({
                "artifact_type": category,
                "artifact_id": f"versions/{os.path.basename(version_dir)}/{folder_name}/{rel_path}",
                "name": art["name"],
                "path": f"{folder_name}/{rel_path}",
                "extension": art["extension"],
                "size_bytes": art["size_bytes"],
                "lines_of_code": art["lines_of_code"],
                "raw_content_sha256": art["raw_content_sha256"],
                "package": art.get("package", ""),
                "imports": art.get("imports", [])
            })
        return written

    def align_commits_to_versions(self, commits: list, version_info: list) -> list:
        """Aligns commits to specific versions based on chronological and topology boundaries."""
        aligned_commits = []
        for c in commits:
            commit_time = datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00"))
            
            # Find matching version by release intervals
            assigned_version = None
            for i, v in enumerate(version_info):
                start_time = version_info[i-1]["release_time"] if i > 0 else datetime.fromisoformat("1970-01-01T00:00:00+00:00")
                end_time = v["release_time"]
                
                if start_time < commit_time <= end_time:
                    assigned_version = v["version"]
                    break
            
            # Fallback to the latest version if commit is newer than latest tag
            if not assigned_version and version_info:
                if commit_time > version_info[-1]["release_time"]:
                    assigned_version = version_info[-1]["version"]
                else:
                    assigned_version = version_info[0]["version"]
                    
            c["associated_versions"] = [assigned_version] if assigned_version else []
            aligned_commits.append(c)
        return aligned_commits

    def align_discussions_to_versions(self, discussions: list, version_info: list) -> list:
        """Aligns issues and pull requests based on their creation and close dates."""
        aligned_discussions = []
        for disc in discussions:
            # Parse created_at/closed_at
            timestamp_str = disc.get("closed_at") or disc.get("created_at")
            if not timestamp_str:
                aligned_discussions.append(disc)
                continue
                
            item_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            assigned_version = None
            
            for i, v in enumerate(version_info):
                start_time = version_info[i-1]["release_time"] if i > 0 else datetime.fromisoformat("1970-01-01T00:00:00+00:00")
                end_time = v["release_time"]
                
                if start_time < item_time <= end_time:
                    assigned_version = v["version"]
                    break
                    
            if not assigned_version and version_info:
                if item_time > version_info[-1]["release_time"]:
                    assigned_version = version_info[-1]["version"]
                else:
                    assigned_version = version_info[0]["version"]
                    
            disc["associated_versions"] = [assigned_version] if assigned_version else []
            aligned_discussions.append(disc)
        return aligned_discussions
