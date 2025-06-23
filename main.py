from soe_vinorm.nsw_detector import CRFNSWDetector
from soe_vinorm.nsw_expander import RuleBasedNSWExpander
from soe_vinorm.text_processor import TextPreprocessor


def main():
    preprocessor = TextPreprocessor()
    detector = CRFNSWDetector()
    text = "ĐT Việt Nam giành huy chương ở nhiều bộ môn"
    tokens = preprocessor(text).split()
    tags = detector.detect(tokens)
    expander = RuleBasedNSWExpander()
    result = expander.expand(tokens, tags)

    print(text)
    print(tokens)
    print(tags)
    print(result)


if __name__ == "__main__":
    main()
