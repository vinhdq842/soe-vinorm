from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from soe_vinorm.nsw_detector import CRFNSWDetector
from soe_vinorm.nsw_expander import RuleBasedNSWExpander
from soe_vinorm.text_processor import TextPreprocessor
from soe_vinorm.utils import load_abbreviation_dict, load_vietnamese_syllables


class Normalizer(ABC):
    """
    Abstract base class for text normalizers.
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Normalize text to spoken form.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text in spoken form.
        """
        ...

    @abstractmethod
    def batch_normalize(self, texts: List[str], n_jobs: int = 1) -> List[str]:
        """
        Normalize multiple texts efficiently.

        Args:
            texts: List of input texts to normalize.
            n_jobs: Number of jobs to run in parallel.

        Returns:
            List of normalized texts.
        """
        ...


class SoeNormalizer(Normalizer):
    """
    Effective Vietnamese text normalizer.
    """

    def __init__(
        self,
        vn_dict: Optional[List[str]] = None,
        abbr_dict: Optional[Dict[str, List[str]]] = None,
        nsw_model_path: Optional[str] = None,
    ):
        """
        Initialize the effective Vietnamese normalizer.

        Args:
            vn_dict: List of Vietnamese words for dictionary lookup. If None, use default Vietnamese dictionary.
            abbr_dict: Dictionary of abbreviations and their expansions. If None, use default abbreviation dictionary.
            nsw_model_path: Path to NSW detection model. If None, use default NSW detection model.
        """
        self._vn_dict = vn_dict or load_vietnamese_syllables()
        self._abbr_dict = abbr_dict or load_abbreviation_dict()

        self._preprocessor = TextPreprocessor(self._vn_dict)
        self._nsw_detector = CRFNSWDetector(
            model_path=nsw_model_path,
            vn_dict=self._vn_dict,
            abbr_dict=self._abbr_dict,
        )
        self._nsw_expander = RuleBasedNSWExpander(
            vn_dict=self._vn_dict,
            abbr_dict=self._abbr_dict,
        )

    def normalize(self, text: str) -> str:
        """
        Normalize text to spoken form.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text in spoken form.
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        tokens = self._preprocessor(text).split()

        if not tokens:
            return text.strip()

        nsw_tags = self._nsw_detector.detect(tokens)
        expanded_tokens = self._nsw_expander.expand(tokens, nsw_tags)

        return " ".join(expanded_tokens)

    def batch_normalize(self, texts: List[str], n_jobs: int = 1) -> List[str]:
        """
        Normalize multiple texts efficiently (optimization in-progress).

        Args:
            texts: List of input texts to normalize.
            n_jobs: Number of jobs to run in parallel.

        Returns:
            List of normalized texts.
        """
        if not isinstance(texts, list) or not all(
            isinstance(text, str) for text in texts
        ):
            raise TypeError("Input must be a list of strings")

        if n_jobs <= 0:
            raise ValueError("Number of jobs must be greater than 0")

        if n_jobs == 1:
            return [self.normalize(text) for text in texts]

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            tokenized_texts = [
                text.split() for text in executor.map(self._preprocessor, texts)
            ]
            nsw_tags = self._nsw_detector.batch_detect(tokenized_texts)
            expanded_texts = [
                " ".join(expanded_tokens)
                for expanded_tokens in executor.map(
                    self._nsw_expander.expand, tokenized_texts, nsw_tags
                )
            ]

        return expanded_texts


def normalize_text(
    text: str,
    vn_dict: Optional[List[str]] = None,
    abbr_dict: Optional[Dict[str, List[str]]] = None,
    nsw_model_path: Optional[str] = None,
) -> str:
    """
    Quick normalization function.

    Args:
        text: Input text to normalize.
        vn_dict: Optional Vietnamese dictionary. If None, use default Vietnamese dictionary.
        abbr_dict: Optional abbreviation dictionary. If None, use default abbreviation dictionary.
        nsw_model_path: Path to NSW detection model. If None, use default NSW detection model.

    Returns:
        Normalized text.
    """
    normalizer = SoeNormalizer(
        vn_dict=vn_dict,
        abbr_dict=abbr_dict,
        nsw_model_path=nsw_model_path,
    )
    return normalizer.normalize(text)


def batch_normalize_texts(
    texts: List[str],
    vn_dict: Optional[List[str]] = None,
    abbr_dict: Optional[Dict[str, List[str]]] = None,
    nsw_model_path: Optional[str] = None,
    n_jobs: int = 1,
) -> List[str]:
    """
    Batch normalization function (optimization in-progress).

    Args:
        texts: List of input texts to normalize.
        vn_dict: Optional Vietnamese dictionary. If None, use default Vietnamese dictionary.
        abbr_dict: Optional abbreviation dictionary. If None, use default abbreviation dictionary.
        nsw_model_path: Path to NSW detection model. If None, use default NSW detection model.
        n_jobs: Number of jobs to run in parallel.

    Returns:
        List of normalized texts.
    """
    normalizer = SoeNormalizer(
        vn_dict=vn_dict,
        abbr_dict=abbr_dict,
        nsw_model_path=nsw_model_path,
    )
    return normalizer.batch_normalize(texts, n_jobs=n_jobs)
