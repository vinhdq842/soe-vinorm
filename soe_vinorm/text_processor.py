"""
Vietnamese Text Preprocessor Module

This module provides functionality for preprocessing Vietnamese text,
including tokenization, cleaning, and normalization.
"""

import re
import unicodedata
from typing import List, Tuple, Optional

import unidecode


class TextPreprocessor:
    """
    A class for preprocessing Vietnamese text.
    
    This class provides methods to clean, tokenize, and normalize Vietnamese text
    for further processing or analysis.
    """
    
    # Regular expressions for text processing
    _NON_SPOKEN_CHARS = r"[^-⇒°…€¥₫₭฿₹₽₱—–!#$%&\\*+,./:;<=>?@\[\]^_`~a-zA-Z0-9À-ÃÈ-ÊÌÍÒ-ÕÙÚÝà-ãè-êìíò-õùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ\s]"
    _PUNCTUATION_TOKENS = r"\s*([$€¥₫₭฿₹₽₱!⇒°<>…;!\\|])\s*"
    _NUMERIC_PATTERN = r"^[0-9%:.,/-]+$"
    _RANGE_PATTERN = r"^([0-9.]+(,[0-9]+)?)([-~][0-9.]+(,[0-9]+)?)*$"
    _URL_PATTERN = r"((https?|ftp)://)?(www)?.*(\.(com|org|net|int|edu|gov|mil|co|tk|io|us|tv|ws|fr|de|cn|vn|biz|blog|fr|lol|cc|cf|ga|ml|info)).*"
    _VIETNAMESE_CAPITAL_PATTERN = r"([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý]+)+"
    _VIETNAMESE_CHARS = r"[À-ÃÈ-ÊÌÍÒ-ÕÙÚÝà-ãè-êìíò-õùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ]"
    _VIETNAMESE_CAPITAL_SPLIT = r"([A-ZÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ][a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý]+)"
    _PUNCTUATION_SPLIT = r"([-&.,:?<>=#%\\])"
    _PUNCTUATION_NUMBER_PATTERN = r"([a-zA-Z]*[À-ÃÈ-ÊÌÍÒ-ÕÙÚÝà-ãè-êìíò-õùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ][a-zA-Z]*)([+:,.&?]*)([0-9])"
    _NUMBER_PUNCTUATION_PATTERN = r"([0-9])([+:,.&?]*)([a-zA-Z]*[À-ÃÈ-ÊÌÍÒ-ÕÙÚÝà-ãè-êìíò-õùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ][a-zA-Z]*)"
    _MULTIPLE_DOTS = r"(\s*\.\s*){3,}"
    _NUMBER_SEPARATOR = r"(?<=[0-9])\s*([-/])\s*(?=[0-9])"
    _COLON_COMMA_PATTERN = r"(?<=[0-9])\s*([:,])\s*(?=[^0-9]|$)"
    
    def __init__(self, vn_dict: List[str]):
        """
        Initialize the text preprocessor.
        
        Args:
            vn_dict: List of Vietnamese words for dictionary lookup
        """
        self._vn_dict = set(vn_dict)  # Convert to set for faster lookups
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize Vietnamese text.
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            Cleaned and normalized text
        """
        # Step 1: Unicode normalization
        text = self._normalize_unicode(text)
        
        # Step 2: Remove non-spoken characters
        text = self._remove_non_spoken_chars(text)
        
        # Step 3: Separate punctuation tokens
        text = self._separate_punctuation_tokens(text)
        
        # Step 4: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Step 5: Handle Vietnamese character patterns
        text = self._handle_vietnamese_patterns(text)
        
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
    
    def _handle_vietnamese_patterns(self, text: str) -> str:
        """Handle Vietnamese character patterns with numbers and punctuation."""
        # Shape: Vietnamese chars (punct) number
        text = re.sub(self._NUMBER_PUNCTUATION_PATTERN, r"\1 \2 \3", text)
        
        # Shape: number (punct) Vietnamese chars
        text = re.sub(self._PUNCTUATION_NUMBER_PATTERN, r"\1 \2 \3", text)
        
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
    
    def _process_token(self, word: str) -> List[str]:
        """
        Process a single token/word.
        
        Args:
            word: Input word to process
            
        Returns:
            List of processed tokens
        """
        word = word.strip()
        
        # Handle special cases
        if word == "..." or not word:
            return [word] if word else []
        
        # Extract prefix and suffix punctuation
        prefix, word, suffix = self._extract_punctuation(word)
        
        # Process the main word based on its type
        processed_word = self._classify_and_process_word(word)
        
        # Handle special slash patterns
        processed_word = self._handle_slash_patterns(processed_word)
        
        return prefix + processed_word + suffix
    
    def _extract_punctuation(self, word: str) -> Tuple[List[str], str, List[str]]:
        """Extract prefix and suffix punctuation from a word."""
        suffix = []
        prefix = []
        
        # Extract suffix punctuation
        while word and word[-1] in "-.,:?&":
            suffix = [word[-1]] + suffix
            word = word[:-1]
        
        # Extract prefix punctuation (but not for negative numbers)
        while word and word[0] in "-.,:?&" and not re.match(r"-[0-9.,]+", word):
            prefix.append(word[0])
            word = word[1:]
        
        return prefix, word, suffix
    
    def _classify_and_process_word(self, word: str) -> List[str]:
        """Classify and process a word based on its type."""
        # Check if it's a numeric pattern
        if self._is_numeric_pattern(word):
            return [word]
        
        # Check if it's a URL
        if self._is_url_pattern(word):
            return self._process_url(word)
        
        # Check if it's Vietnamese concatenated syllables
        if self._is_vietnamese_concatenated(word):
            return self._process_vietnamese_concatenated(word)
        
        # Check if it contains Vietnamese characters or is all uppercase
        if self._is_vietnamese_or_uppercase(word):
            return self._process_vietnamese_or_uppercase(word)
        
        # Default case: shape numpunctnotnum
        return self._process_default_case(word)
    
    def _is_numeric_pattern(self, word: str) -> bool:
        """Check if word matches numeric patterns."""
        return (re.match(self._NUMERIC_PATTERN, word) or 
                re.match(self._RANGE_PATTERN, word))
    
    def _is_url_pattern(self, word: str) -> bool:
        """Check if word matches URL patterns."""
        return bool(re.match(self._URL_PATTERN, word))
    
    def _is_vietnamese_concatenated(self, word: str) -> bool:
        """Check if word is Vietnamese concatenated syllables."""
        return (re.search(self._VIETNAMESE_CAPITAL_PATTERN, word) and
                "-" not in word and
                self._try_separating(word)[0])
    
    def _is_vietnamese_or_uppercase(self, word: str) -> bool:
        """Check if word contains Vietnamese characters or is all uppercase."""
        return (word.upper() == word and not re.search(r"[-0-9,.]+", word) or
                re.search(self._VIETNAMESE_CHARS, word))
    
    def _process_url(self, word: str) -> List[str]:
        """Process URL patterns."""
        parts = re.split(r"(,)", word)
        result = [parts[0]]
        for part in parts[1:]:
            result.extend(self._process_token(part))
        return result
    
    def _process_vietnamese_concatenated(self, word: str) -> List[str]:
        """Process Vietnamese concatenated syllables."""
        parts = self._try_separating(word)[1]
        if len(parts) > 1:
            result = []
            for part in parts:
                result.extend(self._process_token(part))
            return result
        return parts
    
    def _process_vietnamese_or_uppercase(self, word: str) -> List[str]:
        """Process Vietnamese or uppercase words."""
        return list(filter(len, re.split(self._PUNCTUATION_SPLIT, word)))
    
    def _process_default_case(self, word: str) -> List[str]:
        """Process default case: shape numpunctnotnum."""
        return re.sub(self._COLON_COMMA_PATTERN, r" \1 ", word).split()
    
    def _handle_slash_patterns(self, word_list: List[str]) -> List[str]:
        """Handle special slash patterns in word list."""
        if not word_list:
            return word_list
        
        result = word_list.copy()
        
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
    
    def _try_separating(self, word: str) -> Tuple[bool, List[str]]:
        """
        Try to separate Vietnamese concatenated syllables.
        
        Args:
            word: Word to separate
            
        Returns:
            Tuple of (can_separate, separated_parts)
        """
        parts = list(filter(len, re.split(self._VIETNAMESE_CAPITAL_SPLIT, word)))
        
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