import pytest

from soe_vinorm.text_processor import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""

    def test_init_with_custom_dict(self, vn_dict):
        """Test preprocessor initialization with custom dictionary."""
        preprocessor = TextPreprocessor(vn_dict=vn_dict)
        assert preprocessor._vn_dict == set(vn_dict)

    def test_init_with_default_dict(self):
        """Test preprocessor initialization with default dictionary."""
        preprocessor = TextPreprocessor()
        assert len(preprocessor._vn_dict) > 0

    def test_call_empty_text(self):
        """Test preprocessing of empty text."""
        preprocessor = TextPreprocessor()
        assert preprocessor("") == ""
        assert preprocessor("   ") == ""

    def test_call_simple_text(self, vn_dict):
        """Test preprocessing of simple text."""
        preprocessor = TextPreprocessor(vn_dict=vn_dict)
        text = "anh rất ngại"
        result = preprocessor(text)
        assert result == "anh rất ngại"

    def test_call_complete_pipeline(self, vn_dict):
        """Test complete preprocessing pipeline."""
        preprocessor = TextPreprocessor(vn_dict=vn_dict)
        text = "anh rất ngại!@#$%^&*()"
        result = preprocessor(text)
        assert isinstance(result, str)
        assert "anh" in result
        assert "rất" in result
        assert "ngại" in result
        assert "! @ # $ % ^ & *" in result

    def test_call_with_punctuation(self, vn_dict):
        """Test preprocessing with punctuation."""
        preprocessor = TextPreprocessor(vn_dict=vn_dict)
        text = "anh, rất ngại!"
        result = preprocessor(text)
        assert "," in result
        assert "!" in result

    def test_call_with_whitespace(self, vn_dict):
        """Test preprocessing with excessive whitespace."""
        preprocessor = TextPreprocessor(vn_dict=vn_dict)
        text = "anh   rất    ngại"
        result = preprocessor(text)
        assert result == "anh rất ngại"

    def test_tokenize_examples(self, tokenize_examples):
        """Test preprocessing with examples."""
        preprocessor = TextPreprocessor()

        for input_text, expected_tokens in tokenize_examples:
            result = preprocessor(input_text).split()
            assert result == expected_tokens

    def test_error_handling(self, vn_dict):
        """Test error handling."""
        preprocessor = TextPreprocessor(vn_dict=vn_dict)

        with pytest.raises(TypeError):
            preprocessor(None)

        with pytest.raises(TypeError):
            preprocessor(123)

    def test_performance_with_large_text(self, vn_dict):
        """Test performance with larger text."""
        preprocessor = TextPreprocessor(vn_dict=vn_dict)

        # Test with larger text
        large_text = "anh rất ngại " * 1000
        result = preprocessor(large_text)
        assert isinstance(result, str)
        assert len(result) > 0
