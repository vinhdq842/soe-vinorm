from soe_vinorm.text_processor import TextPreprocessor


def main():
    preprocessor = TextPreprocessor()
    result = preprocessor("ĐT Việt Nam giành huy chương ở nhiều bộ môn")
    print(result)


if __name__ == "__main__":
    main()
