from functools import lru_cache
from pathlib import Path
from typing import Dict, List


def get_data_path() -> Path:
    """Get the path to the data directory."""
    return Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def load_vietnamese_syllables() -> List[str]:
    """Load a dictionary of Vietnamese syllables from a text file."""
    file_path = get_data_path() / "dictionaries" / "vietnamese-syllables.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Vietnamese syllables file not found: {file_path}")

    words = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)

    return words


@lru_cache(maxsize=1)
def load_abbreviation_dict() -> Dict[str, List[str]]:
    """Load a dictionary of abbreviations from a text file."""
    file_path = get_data_path() / "dictionaries" / "abbreviations.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Abbreviation dictionary file not found: {file_path}")

    abbreviations = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                abbrev, expansion = line.split(":", 1)
                if abbrev not in abbreviations:
                    abbreviations[abbrev] = []
                abbreviations[abbrev].extend(expansion.split(","))

    return abbreviations
