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

Basic usage

```python
from soe_vinorm import SoeNormalizer

normalizer = SoeNormalizer()
text = "Từ năm 2021 đến nay, đây là lần thứ 3 Bộ Công an xây dựng thông tư để quy định liên quan đến mẫu hộ chiếu, giấy thông hành."

# Single
result = normalizer.normalize(text)
print(result)
# Output: Từ năm hai nghìn không trăm hai mươi mốt đến nay , đây là lần thứ ba Bộ Công an xây dựng thông tư để quy định liên quan đến mẫu hộ chiếu , giấy thông hành .

# Batch
results = normalizer.batch_normalize([text] * 5, n_jobs=5)
print(results)
```

Quick function usage

```python
from soe_vinorm import normalize_text

text = "1kg dâu 25 quả, giá 700.000 - Trung bình 30.000đ/quả"
result = normalize_text(text)
print(result)
# Output: một ki lô gam dâu hai mươi lăm quả , giá bảy trăm nghìn - Trung bình ba mươi nghìn đồng trên quả
```

```python
from soe_vinorm import batch_normalize_texts

texts = [
    "Công trình cao 7,9 m bằng chất liệu đồng nguyên chất với trọng lượng gần 7 tấn, bệ tượng cao 3,6 m",
    "Một trường ĐH tại TP.HCM ghi nhận số nguyện vọng đăng ký vào trường tăng kỷ lục, với trên 178.000.",
    "Theo phương án của Ban Quản lý dự án đường sắt, Bộ Xây dựng, tuyến đường sắt đô thị Thủ Thiêm - Long Thành có chiều dài khoảng 42 km, thiết kế đường đôi, khổ đường 1.435 mm, tốc độ thiết kế 120 km/giờ.",
    "iPhone 16 Pro hiện có giá 999 USD cho phiên bản bộ nhớ 128 GB, và 1.099 USD cho bản 256 GB. Trong khi đó, mẫu 16 Pro Max có dung lượng khởi điểm 256 GB với giá 1.199 USD.",
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
Original: Công trình cao 7,9 m bằng chất liệu đồng nguyên chất với trọng lượng gần 7 tấn, bệ tượng cao 3,6 m
Normalized: Công trình cao bảy phẩy chín mét bằng chất liệu đồng nguyên chất với trọng lượng gần bảy tấn , bệ tượng cao ba phẩy sáu mét
--------------------------------------------------
Original: Một trường ĐH tại TP.HCM ghi nhận số nguyện vọng đăng ký vào trường tăng kỷ lục, với trên 178.000.
Normalized: Một trường Đại học tại Thành phố Hồ Chí Minh ghi nhận số nguyện vọng đăng ký vào trường tăng kỷ lục , với trên một trăm bảy mươi tám nghìn .
--------------------------------------------------
Original: Theo phương án của Ban Quản lý dự án đường sắt, Bộ Xây dựng, tuyến đường sắt đô thị Thủ Thiêm - Long Thành có chiều dài khoảng 42 km, thiết kế đường đôi, khổ đường 1.435 mm, tốc độ thiết kế 120 km/giờ.
Normalized: Theo phương án của Ban Quản lý dự án đường sắt , Bộ Xây dựng , tuyến đường sắt đô thị Thủ Thiêm - Long Thành có chiều dài khoảng bốn mươi hai ki lô mét , thiết kế đường đôi , khổ đường một nghìn bốn trăm ba mươi lăm mi li mét , tốc độ thiết kế một trăm hai mươi ki lô mét trên giờ .
--------------------------------------------------
Original: iPhone 16 Pro hiện có giá 999 USD cho phiên bản bộ nhớ 128 GB, và 1.099 USD cho bản 256 GB. Trong khi đó, mẫu 16 Pro Max có dung lượng khởi điểm 256 GB với giá 1.199 USD.
Normalized: iPhone mười sáu Pro hiện có giá chín trăm chín mươi chín U Ét Dê cho phiên bản bộ nhớ một trăm hai mươi tám ghi ga bai , và một nghìn không trăm chín mươi chín U Ét Dê cho bản hai trăm năm mươi sáu ghi ga bai . Trong khi đó , mẫu mười sáu Pro Max có dung lượng khởi điểm hai trăm năm mươi sáu ghi ga bai với giá một nghìn một trăm chín mươi chín U Ét Dê .
--------------------------------------------------
```

## Approach: Two-stage normalization

### Preprocessing & tokenizing
- Extra spaces, ASCII art, emojis, HTML entities, unspoken words, etc., are removed.
- A regex-based tokenizer is then used to split the sentence into tokens.

### Stage 1: Non-standard word detection
- A sequence tagger is used to extract non-standard words (NSWs) and categorize them into 18 different types.
- These NSWs can later be verbalized properly according to their types.
- The sequence tagger can be any sequence labeling model; this implementation uses a Conditional Random Field due to limited data.

### Stage 2: Non-standard word normalization
- With the NSWs detected in **Stage 1** and their respective types, regex-based expanders are applied to produce normalized results.
- Each NSW type has its own dedicated expander.
- The normalized results are then inserted back into the original sentence, resulting in the desired normalized sentence.

### Other details
- Foreign NSWs are currently kept as-is.
- Abbreviation NSWs
  - **v0.1**: Quantized PhoBERT model combined with a Vietnamese abbreviation dictionary.
  - **v0.2**: A small neural network, also incorporating a Vietnamese abbreviation dictionary.

## Testing
Run all tests with:
```bash
pytest tests
```

## Author
- Vinh Dang (<quangvinh0842@gmail.com>)

## License
MIT License
