import pytest

from soe_vinorm.nsw_detector import CRFNSWDetector


class TestCRFNSWDetector:
    """Test cases for CRFNSWDetector class."""

    def test_init_with_custom_dicts(self, vn_dict, abbr_dict):
        """Test detector initialization with custom dictionaries."""
        detector = CRFNSWDetector(vn_dict=vn_dict, abbr_dict=abbr_dict)
        assert detector._vn_dict == set(vn_dict)
        assert detector._abbr_dict == set(abbr_dict)

    def test_init_with_default_dicts(self):
        """Test detector initialization with default dictionaries."""
        detector = CRFNSWDetector()
        assert len(detector._vn_dict) > 0
        assert len(detector._abbr_dict) > 0

    def test_detect_empty_tokens(self, vn_dict, abbr_dict):
        """Test detection with empty token list."""
        detector = CRFNSWDetector(vn_dict=vn_dict, abbr_dict=abbr_dict)
        result = detector.detect([])
        assert result == []

    def test_detect_simple_example(self, vn_dict, abbr_dict):
        """Test detection with simple example."""
        detector = CRFNSWDetector(vn_dict=vn_dict, abbr_dict=abbr_dict)
        tokens = ["anh", "rất", "ngại"]
        result = detector.detect(tokens)
        assert len(result) == len(tokens)

    def test_detect_examples(self, nsw_detect_examples):
        """Test detection with real examples."""
        detector = CRFNSWDetector()

        for tokens in nsw_detect_examples:
            result = detector.detect(tokens)
            assert len(result) == len(tokens)

    def test_batch_detect_empty_list(self, vn_dict, abbr_dict):
        """Test batch detection with empty list."""
        detector = CRFNSWDetector(vn_dict=vn_dict, abbr_dict=abbr_dict)
        result = detector.batch_detect([])
        assert result == []

    def test_batch_detect_single_text(self, vn_dict, abbr_dict):
        """Test batch detection with single text."""
        detector = CRFNSWDetector(vn_dict=vn_dict, abbr_dict=abbr_dict)
        texts = [["anh", "có", "123", "đồng"]]
        result = detector.batch_detect(texts)
        assert len(result) == 1
        assert len(result[0]) == len(texts[0])

    def test_batch_detect_multiple_texts(self, vn_dict, abbr_dict):
        """Test batch detection with multiple texts."""
        detector = CRFNSWDetector(vn_dict=vn_dict, abbr_dict=abbr_dict)
        texts = [
            ["anh", "có", "123", "đồng"],
            ["ĐT", "Việt", "Nam"],
            ["nhiệt", "độ", "25°C"],
        ]
        result = detector.batch_detect(texts)
        assert len(result) == len(texts)
        assert all(len(tags) == len(texts[i]) for i, tags in enumerate(result))

    def test_batch_detect_consistency(self, vn_dict, abbr_dict):
        """Test that batch detection produces same results as individual detection."""
        detector = CRFNSWDetector(vn_dict=vn_dict, abbr_dict=abbr_dict)
        texts = [["anh", "có", "123", "đồng"], ["ĐT", "Việt", "Nam"]]

        # Individual detection
        individual_results = [detector.detect(tokens) for tokens in texts]

        # Batch detection
        batch_results = detector.batch_detect(texts)

        assert individual_results == batch_results

    def test_feature_extraction(self, vn_dict, abbr_dict):
        """Test feature extraction for tokens."""
        detector = CRFNSWDetector(vn_dict=vn_dict, abbr_dict=abbr_dict)
        tokens = ["anh", "có", "123", "đồng"]
        features = detector._feature_extractor.extract_features(tokens)

        assert len(features) == len(tokens)
        assert all(isinstance(f, dict) for f in features)

    def test_detector_error_handling(self, vn_dict, abbr_dict):
        """Test detector error handling."""
        detector = CRFNSWDetector(vn_dict=vn_dict, abbr_dict=abbr_dict)

        with pytest.raises(TypeError):
            detector.detect(None)

        with pytest.raises(TypeError):
            detector.detect("not a list")

        with pytest.raises(TypeError):
            detector.batch_detect(None)

        with pytest.raises(TypeError):
            detector.batch_detect("not a list")

        with pytest.raises(TypeError):
            detector.batch_detect([["token"], "not a list"])

    def test_detector_performance(self, vn_dict, abbr_dict):
        """Test detector performance with larger inputs."""
        detector = CRFNSWDetector(vn_dict=vn_dict, abbr_dict=abbr_dict)

        # Test with larger token list
        tokens = ["token"] * 1000
        result = detector.detect(tokens)
        assert len(result) == len(tokens)

        # Test batch detection with multiple texts
        texts = [["token"] * 20 for _ in range(1000)]
        batch_result = detector.batch_detect(texts)
        assert len(batch_result) == len(texts)
        assert all(len(tags) == len(texts[i]) for i, tags in enumerate(batch_result))
