"""
Soe Vinorm: Vietnamese Text Normalization Toolkit

A Python library for converting Vietnamese text to its spoken form.
"""

__version__ = "0.1.5"
__author__ = "Vinh Dang"
__email__ = "quangvinh0842@gmail.com"

from .normalizer import (
    SoeNormalizer,
    batch_normalize_texts,
    normalize_text,
)
from .nsw_detector import CRFNSWDetector
from .nsw_expander import RuleBasedNSWExpander
from .text_processor import TextPostprocessor, TextPreprocessor
from .utils import (
    load_abbreviation_dict,
    load_vietnamese_syllables,
)

__all__ = [
    "SoeNormalizer",
    "normalize_text",
    "batch_normalize_texts",
    "CRFNSWDetector",
    "RuleBasedNSWExpander",
    "TextPreprocessor",
    "TextPostprocessor",
    "load_vietnamese_syllables",
    "load_abbreviation_dict",
]
