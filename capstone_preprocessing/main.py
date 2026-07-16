import os
import json
import logging
import argparse
from datetime import datetime

from src.cloner import RepositoryCloner
from src.extractor import ArtifactExtractor
from src.cleaner import DataCleaner
from src.standardizer import MetadataStandardizer
from src.aligner import VersionAligner
from src.relationship import RelationshipPreserver

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="Run MSAI 699 Capstone Preprocessing Pipeline")
    parser.add_argument("--config", default="config.json", help="Path to config.json file")
    return parser.parse_args()

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

def run_pipeline():
    args = parse_args()
    logger.info("Initializing Capstone Preprocessing Pipeline...")
    
    # 1. Load configuration
    config = load_config(args.config)
    repo_url = config.get("repo_url")
    github_repo = config.get("github_repo_identifier")
    github_token = config.get("github_token") or os.environ.get("GITHUB_TOKEN")
    output_dir = config.get("output_dir", "./output")
    target_tags = config.get("version_tags", [])
    
    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_repo_path = os.path.join(script_dir, "work", "spring-petclinic")
    output_base_dir = os.path.join(script_dir, output_dir, "prepared_dataset")
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 2. Clone and Inspect Repository
    cloner = RepositoryCloner(repo_url, local_repo_path)
    repo = cloner.clone_or_open()
    
    available_tags = cloner.get_available_tags()
    logger.info(f"Available tags in repository: {available_tags}")
    
    # If target_tags not defined, fallback to tags or recent commits
    if not target_tags:
        if len(available_tags) >= 2:
            target_tags = available_tags[-3:]
            logger.info(f"No version tags specified. Selected automatically from git tags: {target_tags}")
        else:
            logger.info("Fewer than 2 tags found in repository. Using recent commits as version milestones.")
            # Use three specific commit boundaries from main branch history for testing version evolution
            recent_commits = cloner.get_commit_history(max_count=3)
            target_tags = [rc["hash"] for rc in recent_commits]
            logger.info(f"Selected automatically from commit history: {target_tags}")
    else:
        logger.info(f"Target version tags for research: {target_tags}")
        
    # 3. Setup Aligner and resolve version boundaries
    aligner = VersionAligner(local_repo_path, output_base_dir)
    version_info = aligner.get_tag_info(target_tags)
    
    if not version_info:
        raise ValueError("None of the specified version tags or commit hashes could be resolved in the repository!")
        
    logger.info(f"Version boundaries resolved: {version_info}")
    
    # 4. Initialize extractors, cleaners, standardizers
    extractor = ArtifactExtractor(local_repo_path, github_repo, github_token)
    cleaner = DataCleaner()
    standardizer = MetadataStandardizer()
    
    # 5. Extract Code & Tests per version snapshot (Version Alignment)
    # Define callback to extract files from current working tree
    def extract_current_snapshot():
        raw_files = extractor.extract_local_files()
        cleaned_files = cleaner.clean_artifacts(raw_files)
        return standardized_files
        
    # Let's override the callback to align with extraction logic
    # In order to extract, we checkout a version, list files, clean, and return
    # Let's implement it inside the checkout loop
    file_artifacts_by_version = {}
    
    original_branch = repo.active_branch.name if not repo.head.is_detached else "main"
    try:
        for i, info in enumerate(version_info):
            version = info["version"]
            commit_hash = info["commit_hash"]
            logger.info(f"--- Processing Snapshot for {version} ---")
            
            # Checkout specific release
            repo.git.checkout(commit_hash)
            
            # Extract raw files
            raw_files = extractor.extract_local_files()
            # Clean files
            cleaned_files = cleaner.clean_artifacts(raw_files)
            # Standardize files
            std_files = standardizer.standardize_file_metadata(cleaned_files)
            
            # Write to output/prepared_dataset/versions/<version>/
            version_dir = os.path.join(output_base_dir, "versions", version)
            os.makedirs(version_dir, exist_ok=True)
            
            organized_files = aligner._write_snapshot_files(version_dir, std_files)
            file_artifacts_by_version[version] = organized_files
            
            # Write version metadata
            prev_time = version_info[i-1]["release_time"].isoformat() if i > 0 else "1970-01-01T00:00:00Z"
            meta = {
                "version": version,
                "commit_hash": commit_hash,
                "release_date": info["release_time"].isoformat(),
                "time_boundary_start": prev_time,
                "time_boundary_end": info["release_time"].isoformat(),
                "files_count": len(organized_files)
            }
            with open(os.path.join(version_dir, "version_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
                
            logger.info(f"Completed version {version}. Total files organized: {len(organized_files)}")
            
    finally:
        # Restore git repo HEAD
        logger.info(f"Restoring HEAD to {original_branch}")
        try:
            repo.git.checkout(original_branch)
        except Exception:
            repo.git.checkout("main")
            
    # 6. Extract commits and parse file changes
    logger.info("Extracting Git commits and matching file changes...")
    raw_commits = cloner.get_commit_history(max_count=300) # limit to keep runtime short
    
    # Add files_changed metadata to commits
    # To keep it fast, we parse diffs for the latest 300 commits
    for rc in raw_commits:
        h = rc["hash"]
        commit_obj = repo.commit(h)
        rc["files_changed"] = []
        try:
            # Check diff against parent
            parent = commit_obj.parents[0] if commit_obj.parents else None
            diffs = parent.diff(commit_obj) if parent else commit_obj.diff(None, create_patch=False)
            for diff in diffs:
                rc["files_changed"].append({
                    "path": diff.b_path or diff.a_path,
                    "change_type": "added" if diff.new_file else "deleted" if diff.deleted_file else "modified",
                    "additions": 0,  # GitPython stats can be queried, but diff objects are faster
                    "deletions": 0
                })
        except Exception as e:
            logger.warning(f"Error parsing diff for commit {h[:8]}: {e}")
            
    # Standardize and Align commits
    std_commits = standardizer.standardize_commit_metadata(raw_commits)
    aligned_commits = aligner.align_commits_to_versions(std_commits, version_info)
    
    # 7. Extract issues and PRs (Version Alignment)
    logger.info("Extracting GitHub discussions (Issues & PRs)...")
    raw_issues, raw_prs = extractor.fetch_github_issues_and_prs()
    
    cleaned_issues = cleaner.clean_discussions(raw_issues)
    cleaned_prs = cleaner.clean_discussions(raw_prs)
    
    std_issues = standardizer.standardize_discussion_metadata(cleaned_issues, "issue")
    std_prs = standardizer.standardize_discussion_metadata(cleaned_prs, "pull_request")
    
    aligned_issues = aligner.align_discussions_to_versions(std_issues, version_info)
    aligned_prs = aligner.align_discussions_to_versions(std_prs, version_info)
    
    # 8. Relationship Mapping (Traceability preservation)
    logger.info("Mapping and preserving relationships among artifacts...")
    rel_preserver = RelationshipPreserver()
    relationships = rel_preserver.identify_relationships(
        file_artifacts=file_artifacts_by_version,
        commits=aligned_commits,
        issues=aligned_issues,
        prs=aligned_prs
    )
    
    # 9. Save global dataset files
    logger.info("Saving global dataset indexes...")
    with open(os.path.join(output_base_dir, "commits.json"), "w") as f:
        json.dump(aligned_commits, f, indent=2)
        
    with open(os.path.join(output_base_dir, "issues.json"), "w") as f:
        json.dump(aligned_issues, f, indent=2)
        
    with open(os.path.join(output_base_dir, "pull_requests.json"), "w") as f:
        json.dump(aligned_prs, f, indent=2)
        
    with open(os.path.join(output_base_dir, "relationships.json"), "w") as f:
        json.dump(relationships, f, indent=2)
        
    # Write dataset statistics summary
    dataset_info = {
        "project_name": "Spring PetClinic",
        "repository_url": repo_url,
        "extraction_timestamp": datetime.now().isoformat(),
        "versions_processed": target_tags,
        "statistics": {
            "total_versions": len(target_tags),
            "total_commits": len(aligned_commits),
            "total_issues": len(aligned_issues),
            "total_prs": len(aligned_prs),
            "total_relationships": len(relationships)
        }
    }
    with open(os.path.join(output_base_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
        
    logger.info("=== Pipeline Completed Successfully ===")
    logger.info(f"Dataset stats: {dataset_info['statistics']}")
    logger.info(f"Output files stored in: {output_base_dir}")

if __name__ == "__main__":
    run_pipeline()
