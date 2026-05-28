import json
import re
from pathlib import Path
from typing import Optional
from shared.schemas import JobDescription


def clean_text(text: str) -> str:
    """Clean text: strip HTML, normalize whitespace, lowercase."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing
    text = text.strip()
    return text


def load_job_postings(n: int = 10, data_dir: Optional[str] = None) -> list[JobDescription]:
    """Load sample job postings from JSON files.

    Args:
        n: Number of postings to load (returns up to n)
        data_dir: Directory containing JSON files (defaults to ./sample_data/)

    Returns:
        List of JobDescription objects
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "sample_data"
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        return []

    postings = []
    json_files = sorted(data_dir.glob("*.json"))[:n]

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                posting = JobDescription(**data)
                postings.append(posting)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse {json_file.name}: {e}")

    return postings


def load_resume(resume_path: str) -> dict:
    """Load a resume from a JSON file.

    Args:
        resume_path: Path to resume JSON file

    Returns:
        Dict with 'raw_text' and 'sections' keys
    """
    try:
        with open(resume_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading resume: {e}")
        return {"raw_text": "", "sections": {}}
