"""
Vietnamese Text Normalization Tool

A Python library for normalizing Vietnamese text to spoken form.
"""

__version__ = "0.1.0"
__author__ = "Vinh Dang"
__email__ = "quangvinh0842@gmail.com"

# Main normalizer classes
from .normalizer import (
    batch_normalize_texts,
    normalize_text,
)
from .nsw_detector import CRFNSWDetector, NSWDetector
from .nsw_expander import NSWExpander, RuleBasedNSWExpander

# Individual components
from .text_processor import TextPostprocessor, TextPreprocessor

# Utility functions
from .utils import (
    get_data_path,
    load_abbreviation_dict,
    load_vietnamese_syllables,
)

__all__ = [
    # Main normalizer
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
