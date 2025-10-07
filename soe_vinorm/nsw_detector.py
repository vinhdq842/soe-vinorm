import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Set, Union

from sklearn_crfsuite import CRF

from soe_vinorm.constants import MEASUREMENT_UNITS_MAPPING, MONEY_UNITS_MAPPING
from soe_vinorm.utils import (
    get_model_weights_path,
    load_abbreviation_dict,
    load_vietnamese_syllables,
)


class NSWDetector(ABC):
    """
    Abstract base class for Non-Standard Word (NSW) detectors.
    """

    @abstractmethod
    def detect(self, tokenized_text: List[str]) -> List[str]:
        """Detect NSW labels for a single tokenized text."""
        ...

    @abstractmethod
    def batch_detect(self, tokenized_texts: List[List[str]]) -> List[List[str]]:
        """Detect NSW labels for multiple tokenized texts."""
        ...

    def get_labels(self) -> List[str]:
        """Get the list of possible labels."""
        return ["NSW", "NOT_NSW"]


class CRFNSWDetector(NSWDetector):
    """
    NSW detector using Conditional Random Fields (CRF) model.
    """

    class _FeatureExtractor:
        """
        Feature extractor for CRF NSW detection.
        """

        def __init__(self, vn_dict: Set[str], abbr_dict: Set[str]):
            """
            Initialize the feature extractor.

            Args:
                vn_dict: Set of Vietnamese words for dictionary lookup.
                abbr_dict: Set of abbreviations for abbreviation lookup.
            """
            self._vn_dict = vn_dict
            self._abbr_dict = abbr_dict

        def extract_features(self, tokenized_text: List[str]) -> List[Dict[str, Any]]:
            """
            Extract features for all tokens in the text.

            Args:
                tokenized_text: List of tokens to extract features from.

            Returns:
                List of feature dictionaries for each token.
            """
            return [
                self._extract_token_features(tokenized_text, i)
                for i in range(len(tokenized_text))
            ]

        def _extract_token_features(
            self, tokenized_text: List[str], index: int
        ) -> Dict[str, Any]:
            """
            Extract features for a single token.

            Args:
                tokenized_text: List of all tokens.
                index: Index of the current token.

            Returns:
                Dictionary of features for the token.
            """
            token = tokenized_text[index]

            features = {
                # Basic token information
                "wi": token,
                "is_first_capital": token[0].isupper(),
                "is_first_word": index == 0,
                "is_last_word": index == len(tokenized_text) - 1,
                "is_complete_capital": token.upper() == token,
                "is_alphanumeric": self._is_alphanumeric(token),
                "is_numeric": token.isdigit(),
                # Context features
                "prev_word": "" if index == 0 else tokenized_text[index - 1],
                "next_word": ""
                if index == len(tokenized_text) - 1
                else tokenized_text[index + 1],
                "prev_word_2": "" if index < 2 else tokenized_text[index - 2],
                "next_word_2": ""
                if index > len(tokenized_text) - 3
                else tokenized_text[index + 2],
                # Morphological features
                "prefix_1": token[0],
                "prefix_2": token[:2],
                "prefix_3": token[:3],
                "prefix_4": token[:4],
                "suffix_1": token[-1],
                "suffix_2": token[-2:],
                "suffix_3": token[-3:],
                "suffix_4": token[-4:],
                # Word shape features
                "ws": self._get_word_shape(token),
                "short_ws": self._get_short_word_shape(token),
                # Dictionary lookup features
                "in_vn_dict": self._in_dict(token.lower(), self._vn_dict),
                "in_abbr_dict": self._in_dict(token, self._abbr_dict),
                "in_money_dict": self._in_dict(token, MONEY_UNITS_MAPPING),
                "in_measurement_dict": self._in_dict(token, MEASUREMENT_UNITS_MAPPING),
                # Special character features
                "word_has_hyphen": "-" in token or "–" in token,
                "word_has_tilde": "~" in token,
                "word_has_at": "@" in token,
                "word_has_comma": "," in token,
                "word_has_colon": ":" in token,
                "word_has_dot": "." in token,
                # Pattern features
                "word_has_ws_xxslashxxxx": bool(re.match(r"^\d{1,2}\/\d{4}$", token)),
                "word_has_romanslashxxxx": bool(
                    re.match(r"^[IVXLCDM]+[/.-]\d{4}$", token)
                ),
                "word_has_num_dash_colon_num": bool(
                    re.match(r"^\d[\d.,]*([-–:]\d[\d.,]*)+$", token)
                ),
                "word_contain_only_roman": bool(re.match(r"^[IVXLCDM]+$", token)),
                # Time and date pattern features
                "word_has_time_shape": self._is_time_pattern(token),
                "word_has_day_shape": self._is_day_pattern(token),
                "word_has_date_shape": self._is_date_pattern(token),
                "word_has_month_shape": self._is_month_pattern(token),
            }

            return features

        def _in_dict(self, token: str, dict: Union[Set[str], Dict[str, str]]) -> float:
            """Check if token has any part in dictionary."""
            if token in dict or re.sub(r"[-.]|\d+", "", token) in dict:
                return 1.0

            parts = list(filter(None, re.split(r"[-.]|(\d+)", token)))
            if parts:
                return sum(1 for part in parts if part in dict) / len(parts)

            return 0.0

        def _is_alphanumeric(self, token: str) -> bool:
            """Check if token contains both letters and numbers."""
            return bool(re.match(r"^(?=[^0-9]*[0-9])(?=[^a-zA-Z]*[a-zA-Z])", token))

        def _get_word_shape(self, token: str) -> str:
            """Get the word shape pattern."""
            shapes = []
            for char in token:
                if char.isalpha():
                    shapes.append("X" if char.isupper() else "x")
                elif char.isnumeric():
                    shapes.append("d")
                else:
                    shapes.append(char)
            return "".join(shapes)

        def _get_short_word_shape(self, token: str) -> str:
            """Get the short word shape pattern (collapsed consecutive same shapes)."""
            shapes = []
            for char in token:
                if char.isalpha():
                    shape = "X" if char.isupper() else "x"
                elif char.isnumeric():
                    shape = "d"
                else:
                    shape = char

                if not shapes or shapes[-1] != shape:
                    shapes.append(shape)
            return "".join(shapes)

        def _is_time_pattern(self, token: str) -> bool:
            """Check if token matches time patterns."""
            patterns = [
                r"^([01]?[0-9]|2[0-3])[:hg][0-5]?[0-9][:mp][0-5]?[0-9]$",  # HH[:hg]MM[:mp]SS
                r"^([01]?[0-9]|2[0-3])[:hg][0-5]?[0-9]$",  # HH[:hg]MM
                r"^([01]?[0-9]|2[0-3])[hg]$",  # HH[hg]
                r"^([01]?[0-9]|2[0-3])\s*[-/]\s*([01]?[0-9]|2[0-3])[hg]$",  # HH [-/] HH[hg]
                r"^([01]?[0-9]|2[0-3])[hg]([0-5]?[0-9])?\s*-\s*([01]?[0-9]|2[0-3])[hg]([0-5]?[0-9])?$",  # HH[hg](MM)* [-] HH[hg](MM)*
                r"^([01]?[0-9]|2[0-3]):[0-5]?[0-9]\s*-\s*([01]?[0-9]|2[0-3]):[0-5]?[0-9]$",  # HH[:]MM [-] HH[:]MM
            ]
            return any(re.match(pattern, token) for pattern in patterns)

        def _is_day_pattern(self, token: str) -> bool:
            """Check if token matches day patterns."""
            token = token.replace(".", "/")
            patterns = [
                r"^(0?[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])$",  # DD[/-]MM
                r"^(0?[1-9]|[12][0-9]|3[01])\s*-\s*(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])$",  # DD [-] DD[/]MM
                r"^(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])\s*-\s*(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])$",  # DD[/]MM [-] DD[/]MM
            ]
            return any(re.match(pattern, token) for pattern in patterns)

        def _is_date_pattern(self, token: str) -> bool:
            """Check if token matches date patterns."""
            token = token.replace(".", "/")
            patterns = [
                r"^(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/[1-9]\d{2,3}$",  # DD[/]MM[/]YYYY
                r"^(0?[1-9]|[12][0-9]|3[01])-(0?[1-9]|1[0-2])-[1-9]\d{2,3}$",  # DD[-]MM[-]YYYY
                r"^(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])\s*-\s*(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/[1-9]\d{2,3}$",  # DD[/]MM [-] DD[/]MM[/]YYYY
                r"^(0?[1-9]|[12][0-9]|3[01])\s*-\s*(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/[1-9]\d{2,3}$",  # DD [-] DD[/]MM[/]YYYY
                r"^(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/[1-9]\d{2,3}\s*-\s*(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/[1-9]\d{2,3}$",  # DD[/]MM[/]YYYY [-] DD[/]MM[/]YYYY
            ]
            return any(re.match(pattern, token) for pattern in patterns)

        def _is_month_pattern(self, token: str) -> bool:
            """Check if token matches month patterns."""
            token = token.replace(".", "/")
            patterns = [
                r"^(0?[1-9]|1[0-2])/[1-9]\d{2,3}$",  # MM[/]YYYY
                r"^(0?[1-9]|1[0-2])-[1-9]\d{2,3}$",  # MM[-]YYYY
                r"^(0?[1-9]|1[0-2])\s*-\s*(0?[1-9]|1[0-2])/[1-9]\d{2,3}$",  # MM [-] MM[/]YYYY
                r"^(0?[1-9]|1[0-2])/[1-9]\d{2,3}\s*-\s*(0?[1-9]|1[0-2])/[1-9]\d{2,3}$",  # MM[/]YYYY [-] MM[/]YYYY
            ]
            return any(re.match(pattern, token) for pattern in patterns)

    def __init__(
        self,
        model_path: Union[str, None] = None,
        vn_dict: Union[List[str], None] = None,
        abbr_dict: Union[Dict[str, List[str]], None] = None,
        **kwargs,
    ):
        """
        Initialize the CRF NSW detector.

        Args:
            model_path: Path to the model repository directory. If None, use default path.
            vn_dict: List of Vietnamese words for dictionary lookup. If None, use default Vietnamese dictionary.
            abbr_dict: Dictionary of abbreviations and their expansions. If None, use default abbreviation dictionary.
        """
        super().__init__()

        if model_path is None:
            model_path = get_model_weights_path()
        else:
            model_path = Path(model_path)

        model_path = model_path / "nsw_detector"

        with open(model_path / "crf.pkl", "rb") as f:
            self._crf: CRF = pickle.load(f)

        self._vn_dict = set(vn_dict or load_vietnamese_syllables())
        self._abbr_dict = set(abbr_dict or load_abbreviation_dict())
        self._feature_extractor = self._FeatureExtractor(self._vn_dict, self._abbr_dict)

    def detect(self, tokenized_text: List[str]) -> List[str]:
        """
        Detect NSW labels for a single tokenized text.

        Args:
            tokenized_text: List of tokens to classify.

        Returns:
            List of labels for each token.
        """
        if not isinstance(tokenized_text, list) or not all(
            isinstance(token, str) for token in tokenized_text
        ):
            raise TypeError("tokenized_text must be a list of strings")

        features = [self._feature_extractor.extract_features(tokenized_text)]
        return self._crf.predict(features)[0].tolist()

    def batch_detect(self, tokenized_texts: List[List[str]]) -> List[List[str]]:
        """
        Detect NSW labels for multiple tokenized texts.

        Args:
            tokenized_texts: List of tokenized texts to classify.

        Returns:
            List of label lists for each text.
        """
        if not isinstance(tokenized_texts, list) or not all(
            isinstance(text, list) and all(isinstance(token, str) for token in text)
            for text in tokenized_texts
        ):
            raise TypeError("tokenized_texts must be a list of lists of strings")

        features = [
            self._feature_extractor.extract_features(text) for text in tokenized_texts
        ]
        return self._crf.predict(features).tolist()

    def get_labels(self) -> List[str]:
        """
        Get the list of possible labels for the CRF model.

        Returns:
            List of BIO-style labels.
        """
        # Base labels for different NSW types
        base_labels = [
            "LABB",  # abbreviation
            "LSEQ",  # sequence
            "LWRD",  # foreign word
            "MEA",  # measurement
            "MONEY",  # money
            "NDAT",  # date
            "NDAY",  # day
            "NDIG",  # digit
            "NFRC",  # fraction
            "NMON",  # month
            "NNUM",  # number
            "NPER",  # percentage
            "NQUA",  # quarter
            "NRNG",  # range
            "NSCR",  # score
            "NTIM",  # time
            "NVER",  # version
            "ROMA",  # roman numerals
            "URLE",  # URL and email
            "O",  # other (not NSW)
        ]

        # Convert to BIO format
        bio_labels = []
        for label in base_labels:
            if label != "O":
                bio_labels.extend([f"B-{label}", f"I-{label}"])
        bio_labels.append("O")

        return bio_labels
