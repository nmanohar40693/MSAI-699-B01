import re
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        pass

    def clean_artifacts(self, artifacts: list) -> list:
        """Cleans and normalizes raw properties of extracted artifacts."""
        cleaned = []
        for art in artifacts:
            # Normalize path slashes
            art["path"] = art["path"].replace("\\", "/")
            art["artifact_id"] = art["artifact_id"].replace("\\", "/")

            if "raw_content" in art:
                # Clean source code and text
                content = art["raw_content"]
                
                # Normalize line endings to LF
                content = content.replace("\r\n", "\n").replace("\r", "\n")
                
                # Strip trailing whitespace on each line
                content = "\n".join(line.rstrip() for line in content.splitlines())
                
                # Remove large blocks of empty lines (max 2 consecutive newlines)
                content = re.sub(r"\n{3,}", "\n\n", content)
                
                art["raw_content"] = content
                
            cleaned.append(art)
        logger.info(f"Cleaned and normalized {len(cleaned)} file artifacts.")
        return cleaned

    def clean_discussions(self, discussions: list) -> list:
        """Cleans issue or PR text by stripping HTML comments and excess whitespace."""
        cleaned = []
        for disc in discussions:
            # Strip markdown HTML comments
            if disc.get("body"):
                body = disc["body"]
                body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
                body = body.strip()
                disc["body"] = body
                
            # Clean comments
            for comment in disc.get("comments", []):
                if comment.get("body"):
                    comment["body"] = re.sub(r"<!--.*?-->", "", comment["body"], flags=re.DOTALL).strip()
                    
            cleaned.append(disc)
        return cleaned
