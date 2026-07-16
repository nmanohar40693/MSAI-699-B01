import os
import logging
from git import Repo, GitCommandError

logger = logging.getLogger(__name__)

class RepositoryCloner:
    def __init__(self, repo_url: str, local_path: str):
        self.repo_url = repo_url
        self.local_path = local_path
        self.repo = None

    def clone_or_open(self) -> Repo:
        """Clones the repository if it doesn't exist locally; otherwise opens it."""
        if not os.path.exists(self.local_path):
            logger.info(f"Cloning repository from {self.repo_url} into {self.local_path}...")
            os.makedirs(self.local_path, exist_ok=True)
            self.repo = Repo.clone_from(self.repo_url, self.local_path)
            logger.info("Clone completed successfully.")
        else:
            logger.info(f"Repository already exists at {self.local_path}. Opening...")
            self.repo = Repo(self.local_path)
            
        # Ensure we fetch latest tags/updates
        try:
            logger.info("Fetching updates and tags from origin...")
            self.repo.git.fetch("--tags")
        except GitCommandError as e:
            logger.warning(f"Could not fetch latest tags/updates from origin: {e}")
            
        return self.repo

    def get_available_tags(self):
        """Returns a list of tags sorted by name."""
        if not self.repo:
            raise ValueError("Repository not cloned or opened.")
        return sorted([tag.name for tag in self.repo.tags])

    def checkout_version(self, tag_or_commit: str):
        """Checks out a specific tag or commit hash."""
        if not self.repo:
            raise ValueError("Repository not cloned or opened.")
        logger.info(f"Checking out: {tag_or_commit}")
        try:
            self.repo.git.checkout(tag_or_commit)
        except GitCommandError as e:
            logger.error(f"Failed to checkout {tag_or_commit}: {e}")
            raise e

    def get_commit_history(self, max_count: int = 1000):
        """Retrieves commit metadata from current history."""
        if not self.repo:
            raise ValueError("Repository not cloned or opened.")
        
        commits = []
        # Get commits on active branch/head
        for commit in self.repo.iter_commits(max_count=max_count):
            commits.append({
                "hash": commit.hexsha,
                "author": commit.author.email,
                "timestamp": commit.committed_datetime.isoformat(),
                "message": commit.message.strip(),
                "parents": [p.hexsha for p in commit.parents]
            })
        return commits
