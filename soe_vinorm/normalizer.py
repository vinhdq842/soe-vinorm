"""
Lightweight Vietnamese text normalizer optimized for performance.

This module provides a streamlined interface for text normalization with minimal overhead.
"""

import re
from typing import Dict, List, Optional, Set

from soe_vinorm.nsw_detector import CRFNSWDetector
from soe_vinorm.nsw_expander import RuleBasedNSWExpander
from soe_vinorm.text_processor import TextPreprocessor
from soe_vinorm.utils import load_abbreviation_dict, load_vietnamese_syllables


class VietnameseNormalizer:
    """
    Lightweight Vietnamese text normalizer optimized for performance.
    
    This class provides a streamlined normalization pipeline with minimal overhead.
    """
    
    def __init__(
        self,
        vn_dict: Optional[List[str]] = None,
        abbr_dict: Optional[Dict[str, List[str]]] = None,
        nsw_model_path: Optional[str] = None,
        enable_nsw_detection: bool = True,
        enable_nsw_expansion: bool = True,
    ):
        """
        Initialize the lightweight Vietnamese normalizer.
        
        Args:
            vn_dict: List of Vietnamese words for dictionary lookup.
            abbr_dict: Dictionary of abbreviations and their expansions.
            nsw_model_path: Path to NSW detection model.
            enable_nsw_detection: Whether to enable NSW detection.
            enable_nsw_expansion: Whether to enable NSW expansion.
        """
        # Convert to sets for faster lookups
        self._vn_dict = set(vn_dict) if vn_dict else set(load_vietnamese_syllables())
        self._abbr_dict = abbr_dict or load_abbreviation_dict()
        
        # Initialize components only if needed
        self._preprocessor = TextPreprocessor(list(self._vn_dict))
        self._nsw_detector = None
        self._nsw_expander = None
        
        if enable_nsw_detection:
            self._nsw_detector = CRFNSWDetector(
                model_path=nsw_model_path,
                vn_dict=list(self._vn_dict),
                abbr_dict=self._abbr_dict,
            )
        
        if enable_nsw_expansion:
            self._nsw_expander = RuleBasedNSWExpander(
                vn_dict=list(self._vn_dict),
                abbr_dict=self._abbr_dict,
            )
        
        # Pre-compile regex patterns for performance
        self._whitespace_pattern = re.compile(r'\s+')
        self._empty_check = re.compile(r'^\s*$')
    
    def normalize(self, text: str) -> str:
        """
        Normalize text to spoken form.
        
        Args:
            text: Input text to normalize.
            
        Returns:
            Normalized text in spoken form.
        """
        # Fast empty check
        if not text or self._empty_check.match(text):
            return text
        
        # Preprocess and tokenize
        processed_text = self._preprocessor(text)
        tokens = processed_text.split()
        
        if not tokens:
            return text
        
        # Detect NSW
        if self._nsw_detector:
            nsw_tags = self._nsw_detector.detect(tokens)
        else:
            nsw_tags = ["O"] * len(tokens)
        
        # Expand NSW
        if self._nsw_expander:
            return self._nsw_expander.expand(tokens, nsw_tags)
        else:
            return " ".join(tokens)
    
    def batch_normalize(self, texts: List[str]) -> List[str]:
        """
        Normalize multiple texts efficiently.
        
        Args:
            texts: List of input texts to normalize.
            
        Returns:
            List of normalized texts.
        """
        if not texts:
            return []
        
        # Use list comprehension for better performance
        return [self.normalize(text) for text in texts]
    
    def normalize_fast(self, text: str) -> str:
        """
        Ultra-fast normalization with minimal processing.
        
        Args:
            text: Input text to normalize.
            
        Returns:
            Normalized text.
        """
        if not text or self._empty_check.match(text):
            return text
        
        # Skip preprocessing for speed
        tokens = self._whitespace_pattern.split(text.strip())
        
        if not tokens:
            return text
        
        # Only expand if expander is available
        if self._nsw_expander:
            # Use simple O tags for all tokens
            nsw_tags = ["O"] * len(tokens)
            return self._nsw_expander.expand(tokens, nsw_tags)
        
        return " ".join(tokens)


# Performance-optimized convenience functions
def normalize_text(
    text: str,
    vn_dict: Optional[List[str]] = None,
    abbr_dict: Optional[Dict[str, List[str]]] = None,
    enable_nsw_detection: bool = True,
    enable_nsw_expansion: bool = True,
) -> str:
    """
    Quick normalization function.
    
    Args:
        text: Input text to normalize.
        vn_dict: Optional Vietnamese dictionary.
        abbr_dict: Optional abbreviation dictionary.
        enable_nsw_detection: Whether to enable NSW detection.
        enable_nsw_expansion: Whether to enable NSW expansion.
        
    Returns:
        Normalized text.
    """
    normalizer = VietnameseNormalizer(
        vn_dict=vn_dict,
        abbr_dict=abbr_dict,
        enable_nsw_detection=enable_nsw_detection,
        enable_nsw_expansion=enable_nsw_expansion,
    )
    return normalizer.normalize(text)


def batch_normalize_texts(
    texts: List[str],
    vn_dict: Optional[List[str]] = None,
    abbr_dict: Optional[Dict[str, List[str]]] = None,
    enable_nsw_detection: bool = True,
    enable_nsw_expansion: bool = True,
) -> List[str]:
    """
    Batch normalization function.
    
    Args:
        texts: List of input texts to normalize.
        vn_dict: Optional Vietnamese dictionary.
        abbr_dict: Optional abbreviation dictionary.
        enable_nsw_detection: Whether to enable NSW detection.
        enable_nsw_expansion: Whether to enable NSW expansion.
        
    Returns:
        List of normalized texts.
    """
    normalizer = VietnameseNormalizer(
        vn_dict=vn_dict,
        abbr_dict=abbr_dict,
        enable_nsw_detection=enable_nsw_detection,
        enable_nsw_expansion=enable_nsw_expansion,
    )
    return normalizer.batch_normalize(texts)


def normalize_fast(text: str) -> str:
    """
    Ultra-fast normalization with minimal overhead.
    
    Args:
        text: Input text to normalize.
        
    Returns:
        Normalized text.
    """
    normalizer = VietnameseNormalizer(
        enable_nsw_detection=False,
        enable_nsw_expansion=True,
    )
    return normalizer.normalize_fast(text)