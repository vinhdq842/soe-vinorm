# Soe Vinorm - Vietnamese Text Normalization Toolkit

Soe Vinorm is an effective and extensible toolkit for Vietnamese text normalization, designed for use in Text-to-Speech (TTS) and NLP pipelines. It detects and expands non-standard words (NSWs) such as numbers, dates, abbreviations, and more, converting them into their spoken forms. This project is based on the paper [Non-Standard Vietnamese Word Detection and Normalization for Text-to-Speech](https://arxiv.org/abs/2209.02971).

## Installation

### Option 1: Clone the repository (for development)
```bash
# Clone the repository
git clone https://github.com/vinhdq842/soe-vinorm.git
cd soe-vinorm

# Install dependencies including development dependencies (using uv)
uv sync --dev
```

### Option 2: Install from PyPI
```bash
# Install using uv
uv add soe-vinorm

# Or using pip
pip install soe-vinorm
```

### Option 3: Install from source
```bash
# Install directly from GitHub
uv pip install git+https://github.com/vinhdq842/soe-vinorm.git
```

## Usage

```python
from soe_vinorm import SoeNormalizer

normalizer = SoeNormalizer()
text = 'Từ năm 2021 đến nay, đây là lần thứ 3 Bộ Công an xây dựng thông tư để quy định liên quan đến mẫu hộ chiếu, giấy thông hành.'

result = normalizer.normalize(text)
print(result)
# Output: Từ năm hai nghìn không trăm hai mươi mốt đến nay , đây là lần thứ ba Bộ Công an xây dựng thông tư để quy định liên quan đến mẫu hộ chiếu , giấy thông hành .
```

### Quick function usage
```python
from soe_vinorm import normalize_text

text = "1kg dâu 25 quả, giá 700.000 - Trung bình 30.000đ/quả"
result = normalize_text(text)
print(result)
# Output: một ki lô gam dâu hai mươi lăm quả , giá bảy trăm nghìn - Trung bình ba mươi nghìn đồng trên quả
```

### Batch processing
```python
from soe_vinorm import batch_normalize_texts

texts = [
    "Tôi có 123.456 đồng trong tài khoản",
    "ĐT Việt Nam giành HCV tại SEA Games 32",
    "Nhiệt độ hôm nay là 25°C, ngày 25/04/2014",
    "Tốc độ xe đạt 60km/h trên quãng đường 150km"
]

# Process multiple texts in parallel (4 worker processes)
results = batch_normalize_texts(texts, n_jobs=4)

for original, normalized in zip(texts, results):
    print(f"Original: {original}")
    print(f"Normalized: {normalized}")
    print("-" * 50)
```

Output:
```
Original: Tôi có 123.456 đồng trong tài khoản
Normalized: Tôi có một trăm hai mươi ba nghìn bốn trăm năm mươi sáu đồng trong tài khoản
--------------------------------------------------
Original: ĐT Việt Nam giành HCV tại SEA Games 32
Normalized: đội tuyển Việt Nam giành Huy chương vàng tại SEA Games ba mươi hai
--------------------------------------------------
Original: Nhiệt độ hôm nay là 25°C, ngày 25/04/2014
Normalized: Nhiệt độ hôm nay là hai mươi lăm độ xê , ngày hai mươi lăm tháng bốn năm hai nghìn không trăm mười bốn
--------------------------------------------------
Original: Tốc độ xe đạt 60km/h trên quãng đường 150km
Normalized: Tốc độ xe đạt sáu mươi ki lô mét trên giờ trên quãng đường một trăm năm mươi ki lô mét
--------------------------------------------------
```

## Approach: Two-stage normalization

### Preprocessing & tokenizing
- The extra spaces, ASCII arts, emojis, HTML entities, unspoken words, etc. are removed.
- A Regex-based tokenizer is then used to split the very sentence into tokens.

### Stage 1: Non-standard word detection
- Use a sequence tagger to extract non-standard words (NSWs) and categorize them into different types (18 in total).
- Later, these NSWs can be verbalized properly according to their types.
- The sequence tagger can be any kind of sequence labeling models. This implementation uses Conditional Random Field due to the shortage of data.

### Stage 2: Non-standard word normalization
- With the NSWs detected in **Stage 1** and their respective types, Regex-based expanders are applied to get the normalized results.
- Each NSW type has its own dedicated expander.
- The normalized results are then inserted into the original sentence, resulting in the desired normalized sentence.

### Minor details
- *Foreign* NSWs are kept as is at the moment.
- To expand *Abbreviation* NSWs, a language model is used (i.e. BERT), incorporated with a Vietnamese abbreviation dictionary.
- ...


## Testing
Run all tests with:
```bash
pytest tests
```

## Author
- Vinh Dang (<quangvinh0842@gmail.com>)

## License
MIT License
