import pytest

from soe_vinorm.normalizer import SoeNormalizer, batch_normalize_texts, normalize_text


class TestSoeNormalizer:
    """Test cases for SoeNormalizer class."""

    def test_init_with_custom_dicts(self, vn_dict, abbr_dict):
        """Test normalizer initialization with custom dictionaries."""
        normalizer = SoeNormalizer(vn_dict=vn_dict, abbr_dict=abbr_dict)
        assert normalizer._vn_dict == vn_dict
        assert normalizer._abbr_dict == abbr_dict

    def test_init_with_default_dicts(self):
        """Test normalizer initialization with default dictionaries."""
        normalizer = SoeNormalizer()
        assert len(normalizer._vn_dict) > 0
        assert len(normalizer._abbr_dict) > 0

    def test_normalize_empty_text(self):
        """Test normalization of empty text."""
        normalizer = SoeNormalizer()
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("   ") == ""

    def test_batch_normalize_empty_list(self):
        """Test batch normalization with empty list."""
        normalizer = SoeNormalizer()
        result = normalizer.batch_normalize([])
        assert result == []

    def test_batch_normalize_with_empty_text(self):
        """Test batch normalization with empty text."""
        normalizer = SoeNormalizer()
        result = normalizer.batch_normalize(["", "   "])
        assert result == ["", ""]

    def test_batch_normalize_single_text(self, vn_dict, abbr_dict):
        """Test batch normalization with single text."""
        normalizer = SoeNormalizer(vn_dict=vn_dict, abbr_dict=abbr_dict)
        texts = ["anh có 123 đồng"]
        result = normalizer.batch_normalize(texts)
        assert len(result) == len(texts)

    def test_batch_normalize_multiple_texts(self, vn_dict, abbr_dict):
        """Test batch normalization with multiple texts."""
        normalizer = SoeNormalizer(vn_dict=vn_dict, abbr_dict=abbr_dict)
        texts = ["anh có 123 đồng", "ĐT Việt Nam", "nhiệt độ 25°C"]
        result = normalizer.batch_normalize(texts)
        assert len(result) == len(texts)
        assert all(isinstance(r, str) for r in result)

    def test_batch_normalize_with_parallel(self, vn_dict, abbr_dict):
        """Test batch normalization with parallel processing."""
        normalizer = SoeNormalizer(vn_dict=vn_dict, abbr_dict=abbr_dict)
        texts = ["anh có 123 đồng", "ĐT Việt Nam", "nhiệt độ 25°C", "tốc độ 60km/h"]
        result = normalizer.batch_normalize(texts, n_jobs=2)
        assert len(result) == len(texts)
        assert all(isinstance(r, str) for r in result)

    def test_batch_normalize_consistency(self, vn_dict, abbr_dict):
        """Test that batch normalization produces same results as individual normalization."""
        normalizer = SoeNormalizer(vn_dict=vn_dict, abbr_dict=abbr_dict)
        texts = ["anh có 123 đồng", "ĐT Việt Nam", "nhiệt độ 25°C"]

        individual_results = [normalizer.normalize(text) for text in texts]
        batch_results = normalizer.batch_normalize(texts)

        assert individual_results == batch_results


class TestNormalizeTextFunction:
    """Test cases for normalize_text convenience function."""

    def test_normalize_text_simple(self, vn_dict, abbr_dict):
        """Test normalize_text function with simple text."""
        text = "anh có 123 đồng"
        result = normalize_text(text, vn_dict=vn_dict, abbr_dict=abbr_dict)
        assert isinstance(result, str)

    def test_normalize_text_with_defaults(self):
        """Test normalize_text function with default dictionaries."""
        text = "Tôi có 123 đồng"
        result = normalize_text(text)
        assert isinstance(result, str)

    def test_normalize_text_empty(self):
        """Test normalize_text function with empty text."""
        result = normalize_text("")
        assert result == ""

        result = normalize_text("   ")
        assert result == ""


class TestBatchNormalizeTextsFunction:
    """Test cases for batch_normalize_texts convenience function."""

    def test_batch_normalize_texts_simple(self, vn_dict, abbr_dict):
        """Test batch_normalize_texts function with simple texts."""
        texts = ["anh có 123 đồng", "ĐT Việt Nam"]
        result = batch_normalize_texts(texts, vn_dict=vn_dict, abbr_dict=abbr_dict)
        assert len(result) == len(texts)
        assert all(isinstance(r, str) for r in result)

    def test_batch_normalize_texts_with_defaults(self):
        """Test batch_normalize_texts function with default dictionaries."""
        texts = ["Tôi có 123 đồng", "Nhiệt độ 25°C"]
        result = batch_normalize_texts(texts)
        assert len(result) == len(texts)
        assert all(isinstance(r, str) for r in result)

    def test_batch_normalize_texts_empty(self):
        """Test batch_normalize_texts function with empty list."""
        result = batch_normalize_texts([])
        assert result == []

    def test_batch_normalize_texts_with_empty_text(self):
        """Test batch_normalize_texts function with empty text."""
        result = batch_normalize_texts(["", "   "])
        assert result == ["", ""]

    def test_batch_normalize_texts_with_parallel(self, vn_dict, abbr_dict):
        """Test batch_normalize_texts function with parallel processing."""
        texts = ["anh có 123 đồng", "ĐT Việt Nam", "nhiệt độ 25°C"]
        result = batch_normalize_texts(
            texts, vn_dict=vn_dict, abbr_dict=abbr_dict, n_jobs=2
        )
        assert len(result) == len(texts)
        assert all(isinstance(r, str) for r in result)

    def test_batch_normalize_texts_consistency(self, vn_dict, abbr_dict):
        """Test that batch_normalize_texts produces same results as individual normalize_text."""
        texts = ["anh có 123 đồng", "ĐT Việt Nam"]

        individual_results = [
            normalize_text(text, vn_dict=vn_dict, abbr_dict=abbr_dict) for text in texts
        ]

        batch_results = batch_normalize_texts(
            texts, vn_dict=vn_dict, abbr_dict=abbr_dict, n_jobs=2
        )

        assert individual_results == batch_results


class TestNormalizerIntegration:
    """Integration tests for normalizer components."""

    def test_normalizer_with_real_examples(self, normalize_examples):
        """Test normalizer with real-world examples."""
        normalizer = SoeNormalizer()

        for example in normalize_examples:
            result = normalizer.normalize(example)
            assert isinstance(result, str)

    def test_normalizer_large_texts(self, vn_dict, abbr_dict):
        """Test normalizer with large texts."""
        normalizer = SoeNormalizer(vn_dict=vn_dict, abbr_dict=abbr_dict)

        texts = ["text 1", "text 2", "text 3"] * 1000
        results1 = [normalizer.normalize(text) for text in texts]
        results2 = normalizer.batch_normalize(texts)

        assert results1 == results2

    def test_normalizer_error_handling(self):
        """Test normalizer error handling."""
        normalizer = SoeNormalizer()

        with pytest.raises(TypeError):
            normalizer.normalize(None)

        with pytest.raises(TypeError):
            normalizer.normalize(123)

        with pytest.raises(TypeError):
            normalizer.batch_normalize(None)

        with pytest.raises(TypeError):
            normalizer.batch_normalize([123])

        with pytest.raises(TypeError):
            normalizer.batch_normalize("not a list")

        with pytest.raises(TypeError):
            normalizer.batch_normalize([123, "not a string"])
