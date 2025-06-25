from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from huggingface_hub import snapshot_download


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


HF_MODEL_REPO_ID = "vinhdq842/soe-vinorm"


@lru_cache(maxsize=1)
def get_model_weights_path(cache_dir=None) -> Path:
    """Download and return the local path to the model weights from Hugging Face Hub."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "soe_vinorm"

    local_dir = Path(
        snapshot_download(
            repo_id=HF_MODEL_REPO_ID,
            cache_dir=str(cache_dir),
            allow_patterns=[
                "abbreviation_expander/tokenizer.json",
                "abbreviation_expander/bert.opt.infer.quant.onnx",
                "nsw_detector/*",
            ],
        )
    )

    if not local_dir.is_dir():
        raise FileNotFoundError(
            f"Error downloading model weights from Hugging Face Hub: {HF_MODEL_REPO_ID}"
        )

    return local_dir
