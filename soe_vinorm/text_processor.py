import re
import unicodedata
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import unidecode

from soe_vinorm.utils import load_vietnamese_syllables


class TextProcessor(ABC):
    """
    Abstract base class for text processors.
    """

    @abstractmethod
    def __call__(self, text: str) -> str: ...


class TextPreprocessor(TextProcessor):
    """
    Preprocess Vietnamese texts by:
    - Unicode normalization
    - Removing non-spoken characters
    - Separating punctuation tokens
    - Normalizing whitespace
    - Handling Vietnamese character patterns
    - etc.
    """

    _NON_SPOKEN_CHARS = r"[^-Ωμ⁰£₩⇒°…€¥₫₭฿₹₽₱—–!#$%&\\*+,./:;<=>?@\[\]^_`~a-zA-Z0-9À-ÃÈ-ÊÌÍÒ-ÕÙÚÝà-ãè-êìíò-õùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ\s]"
    _PUNCTUATION_TOKENS = r"\s*([$€¥₫₭฿₹₽₱!⇒°<>…;!\\|])\s*"
    _NUMERIC_PATTERN = r"^[0-9%:.,/-]+$"
    _RANGE_PATTERN = r"^([0-9.]+(,[0-9]+)?)([-~][0-9.]+(,[0-9]+)?)*$"
    _URL_PATTERN = r"((https?|ftp)://)?(www)?.*(\.(com|org|net|int|edu|gov|mil|co|tk|io|us|tv|ws|fr|de|cn|vn|biz|blog|fr|lol|cc|cf|ga|ml|info)).*"
    _VIETNAMESE_CAPITAL_PATTERN = r"([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý]+)+"
    _VIETNAMESE_CAPITAL_SPLIT = r"([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý]+)"
    _VIETNAMESE_CHARS = r"[À-ÃÈ-ÊÌÍÒ-ÕÙÚÝà-ãè-êìíò-õùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ]"
    _PUNCTUATION_SPLIT = r"([-&.,:?<>=#%\\])"
    _VN_CHAR_PUNCT_NUMBER_PATTERN = r"([a-zA-Z]*[À-ÃÈ-ÊÌÍÒ-ÕÙÚÝà-ãè-êìíò-õùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ][a-zA-Z]*)([+:,.&?]*)([0-9])"
    _NUMBER_PUNCT_VN_CHAR_PATTERN = r"([0-9])([+:,.&?]*)([a-zA-Z]*[À-ÃÈ-ÊÌÍÒ-ÕÙÚÝà-ãè-êìíò-õùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ][a-zA-Z]*)"
    _MULTIPLE_DOTS = r"(\s*\.\s*){3,}"
    _NUMBER_SEPARATOR = r"(?<=[0-9])\s*([-/])\s*(?=[0-9])"
    _COLON_COMMA_PATTERN = r"(?<=[0-9])\s*([:,])\s*(?=[^0-9]|$)"

    def __init__(self, vn_dict: Union[List[str], None] = None):
        """
        Initialize the text preprocessor.

        Args:
            vn_dict: Optional list of Vietnamese syllables.
            If not provided, the default Vietnamese syllables will be used.
        """
        self._vn_dict = set(vn_dict) if vn_dict else set(load_vietnamese_syllables())

    def __call__(self, text: str) -> str:
        """Process the input text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # Step 1: Unicode normalization
        text = self._normalize_unicode(text)

        # Step 2: Remove non-spoken characters
        text = self._remove_non_spoken_chars(text)

        # Step 3: Separate punctuation tokens
        text = self._separate_punctuation_tokens(text)

        # Step 4: Normalize whitespace
        text = self._normalize_whitespace(text)

        # Step 5: Handle Vietnamese character-punctuation-number patterns
        text = self._handle_vietnamese_char_num_patterns(text)

        # Step 6: Replace multiple dots
        text = self._replace_multiple_dots(text)

        # Step 7: Process tokens
        text = self._process_all_tokens(text)

        # Step 8: Concatenate numbers separated by specific symbols
        return self._concatenate_number_separators(text)

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        return unicodedata.normalize("NFC", text)

    def _remove_non_spoken_chars(self, text: str) -> str:
        """Remove non-spoken characters from text."""
        return re.sub(self._NON_SPOKEN_CHARS, " ", text)

    def _separate_punctuation_tokens(self, text: str) -> str:
        """Separate punctuation tokens with spaces."""
        return re.sub(self._PUNCTUATION_TOKENS, r" \1 ", text).strip()

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace by removing redundant spaces."""
        return re.sub(r"\s+", " ", text).strip()

    def _handle_vietnamese_char_num_patterns(self, text: str) -> str:
        """Handle Vietnamese character-punctuation-number patterns."""
        # Vietnamese chars (punct) number (punct)
        text = re.sub(self._VN_CHAR_PUNCT_NUMBER_PATTERN, r"\1 \2 \3", text)

        # Number (punct) Vietnamese chars
        text = re.sub(self._NUMBER_PUNCT_VN_CHAR_PATTERN, r"\1 \2 \3", text)

        return text

    def _replace_multiple_dots(self, text: str) -> str:
        """Replace multiple dots with three dots."""
        return re.sub(self._MULTIPLE_DOTS, " ... ", text)

    def _process_all_tokens(self, text: str) -> str:
        """Process all tokens in the text."""
        tokens = text.split()
        processed_tokens = []

        for token in tokens:
            processed_tokens.extend(self._process_token(token))

        return " ".join(processed_tokens)

    def _concatenate_number_separators(self, text: str) -> str:
        """Concatenate numbers separated by specific symbols."""
        return re.sub(self._NUMBER_SEPARATOR, r"\1", text)

    def _process_token(self, token: str) -> List[str]:
        """Process a single token."""
        token = token.strip()

        # Handle special cases
        if token == "..." or not token:
            return [token] if token else []

        # Extract prefix and suffix punctuation
        prefix, token, suffix = self._extract_punctuation(token)

        # Process the main token based on its type
        processed_token = self._classify_and_process_token(token)

        return prefix + processed_token + suffix

    def _extract_punctuation(self, token: str) -> Tuple[List[str], str, List[str]]:
        """Extract prefix and suffix punctuation from a token."""
        suffix = []
        prefix = []

        # Extract suffix punctuation
        while token and token[-1] in "-.,:?&":
            suffix = [token[-1]] + suffix
            token = token[:-1]

        # Extract prefix punctuation (but not for negative numbers)
        while token and token[0] in "-.,:?&" and not re.match(r"-[0-9.,]+", token):
            prefix.append(token[0])
            token = token[1:]

        return prefix, token, suffix

    def _classify_and_process_token(self, token: str) -> List[str]:
        """Classify and process a token based on its type."""
        # Check if it's a numeric or range pattern
        if self._is_numeric_or_range_pattern(token):
            return [token]

        # Check if it's a URL
        if self._is_url_pattern(token):
            return self._process_url(token)

        # Check if it's Vietnamese concatenated syllables with each's first letter is capital
        if self._is_vietnamese_concatenated_with_capital_first(token):
            return self._process_vietnamese_concatenated_with_capital_first(token)

        # Check if it contains Vietnamese characters or is all uppercase and contains no numbers
        if self._is_vietnamese_or_uppercase(token):
            return self._process_vietnamese_or_uppercase(token)

        # Default case: shape num-punct-not num
        return self._process_default_case(token)

    def _is_numeric_or_range_pattern(self, token: str) -> bool:
        """Check if token matches numeric or range patterns."""
        return re.match(self._NUMERIC_PATTERN, token) or re.match(
            self._RANGE_PATTERN, token
        )

    def _is_url_pattern(self, token: str) -> bool:
        """Check if token matches URL patterns."""
        return bool(re.match(self._URL_PATTERN, token))

    def _is_vietnamese_concatenated_with_capital_first(self, token: str) -> bool:
        """Check if token is Vietnamese concatenated syllables with each's first letter is capital."""
        return (
            re.search(self._VIETNAMESE_CAPITAL_PATTERN, token)
            and "-" not in token
            and self._try_separating(token)[0]
        )

    def _is_vietnamese_or_uppercase(self, token: str) -> bool:
        """Check if token contains Vietnamese characters or is all uppercase and contains no numbers."""
        return (
            token.upper() == token
            and not re.search(r"[-0-9,.]+", token)
            or re.search(self._VIETNAMESE_CHARS, token)
        )

    def _process_url(self, token: str) -> List[str]:
        """Process URL patterns."""
        parts = re.split(r"(,)", token)
        result = [parts[0]]
        for part in parts[1:]:
            result.extend(self._process_token(part))
        return result

    def _process_vietnamese_concatenated_with_capital_first(
        self, token: str
    ) -> List[str]:
        """Process Vietnamese concatenated syllables with each's first letter is capital."""
        parts = self._try_separating(token)[1]
        if len(parts) > 1:
            result = []
            for part in parts:
                result.extend(self._process_token(part))
            return result
        return parts

    def _process_vietnamese_or_uppercase(self, token: str) -> List[str]:
        """Process Vietnamese or uppercase tokens that contains no numbers."""
        return self._handle_slash_patterns(
            list(filter(len, re.split(self._PUNCTUATION_SPLIT, token)))
        )

    def _handle_slash_patterns(self, token_list: List[str]) -> List[str]:
        """Handle special slash patterns in token list."""
        if not token_list:
            return token_list

        result = token_list.copy()

        # Handle ending slashes
        if result and result[-1] and result[-1][-1] == "/":
            tmp = []
            while result[-1] and result[-1][-1] == "/":
                result[-1] = result[-1][:-1]
                tmp.append("/")
            result.extend(tmp)

        # Handle starting slashes
        if result and result[0] and result[0][0] == "/":
            tmp = []
            while result[0] and result[0][0] == "/":
                result[0] = result[0][1:]
                tmp.append("/")
            result = tmp + result

        return result

    def _process_default_case(self, token: str) -> List[str]:
        """Process default case: shape num-punct-not num."""
        return re.sub(self._COLON_COMMA_PATTERN, r" \1 ", token).split()

    def _try_separating(self, token: str) -> Tuple[bool, List[str]]:
        """
        Try to separate if found out that the token is Vietnamese concatenated syllables.

        Args:
            token: Token to separate

        Returns:
            Tuple of (can_separate, separated_parts)
        """
        parts = list(filter(len, re.split(self._VIETNAMESE_CAPITAL_SPLIT, token)))

        def check_dict(part: str) -> bool:
            """Check if a part exists in the Vietnamese dictionary."""
            part_lower = part.lower()
            if len(part_lower) > 1:
                return part_lower in self._vn_dict
            elif len(part_lower) == 1:
                return unidecode.unidecode(part_lower) in "ueoai"
            return False

        can_separate = any(check_dict(part) for part in parts)
        return can_separate, parts


class TextPostprocessor(TextProcessor):
    """
    Postprocess Vietnamese texts.
    """

    def __call__(self, text: str) -> str:
        return text
