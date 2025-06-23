"""
Vietnamese Text Normalization Tool

A Python library for normalizing Vietnamese text to spoken form.
"""

__version__ = "0.1.0"
__author__ = "Vinh Dang"
__email__ = "quangvinh0842@gmail.com"

# Main normalizer classes
from .normalizer import (
    Normalizer,
    VietnameseNormalizer,
    normalize_text,
    batch_normalize_texts,
)

# Individual components
from .text_processor import TextPreprocessor, TextPostprocessor
from .nsw_detector import NSWDetector, CRFNSWDetector
from .nsw_expander import NSWExpander, RuleBasedNSWExpander

# Utility functions
from .utils import (
    get_data_path,
    load_vietnamese_syllables,
    load_abbreviation_dict,
)

__all__ = [
    # Main normalizer
    "Normalizer",
    "VietnameseNormalizer",
    "normalize_text",
    "batch_normalize_texts",
    
    # Individual components
    "TextPreprocessor",
    "TextPostprocessor",
    "NSWDetector",
    "CRFNSWDetector",
    "NSWExpander",
    "RuleBasedNSWExpander",
    
    # Utilities
    "get_data_path",
    "load_vietnamese_syllables",
    "load_abbreviation_dict",
]