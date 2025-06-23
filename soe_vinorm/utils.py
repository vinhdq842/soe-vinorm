"""
Utility functions for soe-vinorm package.
"""

import os
from pathlib import Path
from typing import List, Optional


def get_data_path() -> Path:
    """
    Get the path to the data directory.
    
    Returns:
        Path to the data directory
    """
    # Get the directory where this module is located
    current_dir = Path(__file__).parent
    return current_dir / "data"


def load_dictionary(filename: str, encoding: str = "utf-8") -> List[str]:
    """
    Load a dictionary from a text file.
    
    Args:
        filename: Name of the dictionary file (e.g., "vietnamese_dict.txt")
        encoding: File encoding (default: utf-8)
        
    Returns:
        List of words from the dictionary
        
    Raises:
        FileNotFoundError: If the dictionary file doesn't exist
    """
    data_path = get_data_path()
    file_path = data_path / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dictionary file not found: {file_path}")
    
    words = []
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                words.append(line)
    
    return words


def load_vietnamese_dict() -> List[str]:
    """
    Load the default Vietnamese dictionary.
    
    Returns:
        List of Vietnamese words
    """
    return load_dictionary("vietnamese_dict.txt")


def load_second_dict() -> List[str]:
    """
    Load the second dictionary.
    
    Returns:
        List of words from the second dictionary
    """
    return load_dictionary("second_dict.txt")


def get_available_dictionaries() -> List[str]:
    """
    Get list of available dictionary files.
    
    Returns:
        List of available dictionary filenames
    """
    data_path = get_data_path()
    if not data_path.exists():
        return []
    
    return [f.name for f in data_path.glob("*.txt")]


def combine_dictionaries(*filenames: str) -> List[str]:
    """
    Combine multiple dictionaries into one list.
    
    Args:
        *filenames: Names of dictionary files to combine
        
    Returns:
        Combined list of words (duplicates removed)
    """
    all_words = set()
    
    for filename in filenames:
        try:
            words = load_dictionary(filename)
            all_words.update(words)
        except FileNotFoundError:
            print(f"Warning: Dictionary file '{filename}' not found, skipping...")
    
    return sorted(list(all_words))