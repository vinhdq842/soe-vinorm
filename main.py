from soe_vinorm.nsw_detector import CRFNSWDetector
from soe_vinorm.text_processor import TextPreprocessor


def main():
    preprocessor = TextPreprocessor()
    detector = CRFNSWDetector()
    text = "ĐT Việt Nam giành huy chương ở nhiều bộ môn"
    tokens = preprocessor(text).split()
    tags = detector.detect(tokens)

    print(text)
    print(tokens)
    print(tags)


if __name__ == "__main__":
    main()
