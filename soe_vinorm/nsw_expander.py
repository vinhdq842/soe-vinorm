import re
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from tokenizers import Tokenizer
from unidecode import unidecode

from soe_vinorm.constants import (
    MEASUREMENT_UNITS_MAPPING,
    MONEY_UNITS_MAPPING,
    NUMBER_MAPPING,
    NUMBER_UNIT_SINGLE,
    NUMBER_UNIT_TRIPLE,
    SEQUENCE_PHONES_EN_MAPPING,
    SEQUENCE_PHONES_VI_MAPPING,
)
from soe_vinorm.utils import (
    get_model_weights_path,
    load_abbreviation_dict,
    load_vietnamese_syllables,
)


class NSWExpander(ABC):
    """
    Abstract base class for NSW (Non-Standard Word) expanders.
    """

    @abstractmethod
    def expand(self, words: List[str], tags: List[str]) -> List[str]:
        """Expand a list of words based on their tags."""
        ...


class RuleBasedNSWExpander(NSWExpander):
    """
    NSW expander using rule-based approach.
    """

    class NumberExpander:
        """
        Handles expansion of numbers, digits, and numeric patterns.
        """

        def expand_digit(self, digit: str) -> str:
            """Expand a single digit to its spoken form."""
            result = []
            for char in digit:
                if char in NUMBER_MAPPING:
                    result.append(NUMBER_MAPPING[char])
            return " ".join(result)

        def expand_number(self, number: str) -> str:
            """Expand a number to its spoken form."""
            try:
                # Handle sign
                sign = ""
                if number[0] in ["-", "+"]:
                    sign = {"+": "cộng", "-": "trừ"}[number[0]]
                    number = number[1:]

                # Remove separators and unhandlable chars
                number = number.replace(".", "")
                while len(number) > 1 and number[0] == "0" and number[1].isnumeric():
                    number = number[1:]
                number = re.sub(r"[^0-9.,]", "", number)

                # Handle decimal part
                decimal_part = ""
                if "," in number:
                    decimal_part = f"phẩy {self.expand_digit(number.split(',')[-1])}"
                    number = "".join(number.split(",")[:-1])

                # Split into chunks of 3 digits
                chunks = self._split_into_chunks(number)

                # Process each chunk
                result_parts = []
                for i, chunk in enumerate(chunks):
                    part = self._speak_out_chunk(chunk, len(chunks) - i - 1)
                    if part:
                        result_parts.append(part)

                result = " ".join(result_parts)

                # Apply Vietnamese number pronunciation rules
                result = self._apply_vietnamese_rules(result)

                return (sign + " " + result + " " + decimal_part).strip()

            except IndexError:
                return self.expand_digit(number)

        def _split_into_chunks(self, number: str) -> List[str]:
            """Split number into chunks of 3 digits."""
            chunks = [number[i : i + 3] for i in range(len(number) - 3, -1, -3)][::-1]
            if len(number) % 3:
                chunks = [number[: len(number) % 3]] + chunks
            return chunks

        def _speak_out_chunk(self, chunk: str, unit_index: int) -> str:
            """Convert a number chunk to its spoken form."""
            if chunk == "000":
                return ""

            result = ""
            position = len(chunk) - 1
            while position >= 0:
                if (
                    position == len(chunk) - 1
                    and chunk[position] == "0"
                    and len(chunk) > 1
                ):
                    ...  # Skip trailing zeros
                elif position == len(chunk) - 2 and chunk[position] in ["1", "0"]:
                    if position == 0 and chunk[position] == "0":
                        ...  # Skip leading zeros
                    elif chunk[position] == "1":
                        result = (
                            f"mười {NUMBER_MAPPING[chunk[position + 1]]}"
                            if chunk[position + 1] != "0"
                            else "mười"
                        )
                    else:
                        result = (
                            "linh " + NUMBER_MAPPING[chunk[position + 1]]
                            if chunk[position + 1] != "0"
                            else ""
                        )
                else:
                    result = (
                        NUMBER_MAPPING[chunk[position]]
                        + " "
                        + NUMBER_UNIT_SINGLE[len(chunk) - position - 1]
                        + (" " + result if result else "")
                    )

                position -= 1

            if unit_index >= len(NUMBER_UNIT_TRIPLE):
                raise IndexError("unit_index is too large")

            return " ".join([result.strip(), NUMBER_UNIT_TRIPLE[unit_index]]).strip()

        def _apply_vietnamese_rules(self, text: str) -> str:
            """Apply Vietnamese number pronunciation rules."""
            return (
                text.replace("mười năm", "mười lăm")
                .replace("mươi năm", "mươi lăm")
                .replace("mươi bốn", "mươi tư")
                .replace("mươi một", "mươi mốt")
                .replace("linh bốn", "linh tư")
            )

        def expand_roma(self, roma: str) -> str:
            """Expand a Roman numeral to its spoken form."""
            return self.expand_number(self._roman_to_int(roma))

        def _roman_to_int(self, roman: str) -> str:
            """Convert Roman numerals to integer."""
            roman = roman.strip().upper()
            roman = re.sub(r"[^IVXLCDM]", "", roman)
            number = 0
            subtract = 0

            roman_values = {
                "I": 1,
                "V": 5,
                "X": 10,
                "L": 50,
                "C": 100,
                "D": 500,
                "M": 1000,
            }
            subtract_pairs = {"I": ["V", "X"], "X": ["L", "C"], "C": ["D", "M"]}

            for i, char in enumerate(roman):
                if (
                    char in subtract_pairs
                    and i + 1 < len(roman)
                    and roman[i + 1] in subtract_pairs[char]
                ):
                    subtract += roman_values[char]
                    continue
                number += roman_values[char]

            return str(number - subtract)

    class TimeDateExpander:
        """Handles expansion of time and date patterns."""

        def __init__(self, number_expander, sequence_expander):
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander

        def expand_time(self, time_str: str) -> str:
            """Expand time patterns to spoken form."""
            time_str = time_str.strip()

            # HH[:hg]MM[:mp]SS
            if re.match(r"^\d{1,2}[:hg]\d{1,2}[:mp]\d{1,2}$", time_str):
                h, m, s = re.split(r"[:hgmp]", time_str)[:3]
                return (
                    f"{self._number_expander.expand_number(h)} giờ "
                    f"{self._number_expander.expand_number(m)} phút "
                    f"{self._number_expander.expand_number(s)} giây"
                )
            # HH[:hg]MM
            elif re.match(r"^\d{1,2}[:hg]\d{1,2}$", time_str):
                h, m = re.split(r"[:hg]", time_str)[:2]
                return (
                    f"{self._number_expander.expand_number(h)} giờ "
                    f"{self._number_expander.expand_number(m)} phút"
                )
            # HH[hg]
            elif re.match(r"^\d{1,2}[hg]$", time_str):
                h = re.split(r"[hg]", time_str)[0]
                return f"{self._number_expander.expand_number(h)} giờ"
            # HH [/] HH[hg]
            elif re.match(r"^\d{1,2}\s*/\s*\d{1,2}[hg]$", time_str):
                h1, h2 = re.split(r"\s*/\s*", time_str)[:2]
                return (
                    f"{self._number_expander.expand_number(h1)} trên "
                    f"{self._number_expander.expand_number(h2)} giờ"
                )
            # HH [-] HH[hg]
            elif re.match(r"^\d{1,2}\s*-\s*\d{1,2}[hg]$", time_str):
                h1, h2 = re.split(r"\s*-\s*", time_str)[:2]
                return (
                    f"{self._number_expander.expand_number(h1)} đến "
                    f"{self.expand_time(h2)}"
                )
            # HH[hg] [-] HH[hg]
            elif re.match(r"^\d{1,2}[hg]\s*-\s*\d{1,2}[hg]$", time_str):
                h1, h2 = re.split(r"\s*-\s*", time_str)[:2]
                return f"{self.expand_time(h1)} đến {self.expand_time(h2)}"
            # HH[:hg]MM [-] HH[:hg]MM
            elif re.match(r"^\d{1,2}[:hg]\d{1,2}\s*-\s*\d{1,2}[:hg]\d{1,2}$", time_str):
                t1, t2 = re.split(r"\s*-\s*", time_str)[:2]
                return f"{self.expand_time(t1)} đến {self.expand_time(t2)}"

            return self._sequence_expander.expand_sequence(time_str)

        def expand_day(self, day_str: str) -> str:
            """Expand day patterns to spoken form."""
            day_str = day_str.strip().replace(".", "/")

            # DD[/-]MM
            if re.match(r"^\d{1,2}\s*[/-]\s*\d{1,2}$", day_str):
                d, m = re.split(r"\s*/\s*|\s*-\s*", day_str)[:2]
                prefix = "mùng " if len(d.lstrip("0")) == 1 or d == "10" else ""
                return (
                    f"{prefix}{self._number_expander.expand_number(d)} tháng "
                    f"{self._number_expander.expand_number(m)}"
                )
            # DD [-] DD[/]MM
            elif re.match(r"^\d{1,2}\s*-\s*\d{1,2}/\d{1,2}$", day_str):
                d1, d2 = re.split(r"\s*-\s*", day_str)[:2]
                prefix = "mùng " if len(d1.lstrip("0")) == 1 or d1 == "10" else ""
                return (
                    f"{prefix}{self._number_expander.expand_number(d1)} đến ngày "
                    f"{self.expand_day(d2)}"
                )
            # DD[/]MM [-] DD[/]MM
            elif re.match(r"^\d{1,2}/\d{1,2}\s*-\s*\d{1,2}/\d{1,2}$", day_str):
                d1, d2 = re.split(r"\s*-\s*", day_str)[:2]
                return f"{self.expand_day(d1)} đến ngày {self.expand_day(d2)}"

            return self._sequence_expander.expand_sequence(day_str)

        def expand_date(self, date_str: str) -> str:
            """Expand date patterns to spoken form."""
            date_str = date_str.strip().replace(".", "/")

            # DD[/]MM[/]YYYY
            if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", date_str):
                d, m, y = re.split(r"/", date_str)[:3]
                prefix = "mùng " if len(d.lstrip("0")) == 1 or d == "10" else ""
                return (
                    f"{prefix}{self._number_expander.expand_number(d)} tháng "
                    f"{self._number_expander.expand_number(m)} năm "
                    f"{self._number_expander.expand_number(y)}"
                )
            # DD[-]MM[-]YYYY
            elif re.match(r"^\d{1,2}-\d{1,2}-\d{2,4}$", date_str):
                return self.expand_date(re.sub(r"-", "/", date_str))
            # DD[/]MM [-] DD[/]MM[/]YYYY
            elif re.match(r"^\d{1,2}/\d{1,2}\s*-\s*\d{1,2}/\d{1,2}/\d{2,4}$", date_str):
                d1, d2 = re.split(r"\s*-\s*", date_str)[:2]
                return f"{self.expand_day(d1)} đến ngày {self.expand_date(d2)}"
            # DD [-] DD[/]MM[/]YYYY
            elif re.match(r"^\d{1,2}\s*-\s*\d{1,2}/\d{1,2}/\d{2,4}$", date_str):
                d1, d2 = re.split(r"\s*-\s*", date_str)[:2]
                prefix = "mùng " if len(d1.lstrip("0")) == 1 or d1 == "10" else ""
                return (
                    f"{prefix}{self._number_expander.expand_number(d1)} đến ngày "
                    f"{self.expand_date(d2)}"
                )
            # DD[/]MM[/]YYYY [-] DD[/]MM[/]YYYY
            elif re.match(
                r"^\d{1,2}/\d{1,2}/\d{2,4}\s*-\s*\d{1,2}/\d{1,2}/\d{2,4}$", date_str
            ):
                d1, d2 = re.split(r"\s*-\s*", date_str)[:2]
                return f"{self.expand_date(d1)} đến ngày {self.expand_date(d2)}"

            return self._sequence_expander.expand_sequence(date_str)

        def expand_month(self, month_str: str) -> str:
            """Expand month patterns to spoken form."""
            month_str = month_str.strip().replace(".", "/")

            # MM[/]YYYY
            if re.match(r"^\d{1,2}/\d{2,4}$", month_str):
                m, y = month_str.split("/")[:2]
                return (
                    f"{self._number_expander.expand_number(m)} năm "
                    f"{self._number_expander.expand_number(y)}"
                )
            # MM[-]YYYY
            elif re.match(r"^\d{1,2}-\d{2,4}$", month_str):
                return self.expand_month(month_str.replace("-", "/"))
            # MM [-] MM[/]YYYY
            elif re.match(r"^\d{1,2}\s*-\s*\d{1,2}/\d{2,4}$", month_str):
                p1, p2 = re.split(r"\s*-\s*", month_str)[:2]
                return (
                    f"{self._number_expander.expand_number(p1)} đến tháng "
                    f"{self.expand_month(p2)}"
                )
            # MM[/]YYYY [-] MM[/]YYYY
            elif re.match(r"^\d{1,2}/\d{2,4}\s*-\s*\d{1,2}/\d{2,4}$", month_str):
                p1, p2 = re.split(r"\s*-\s*", month_str)[:2]
                return f"{self.expand_month(p1)} đến tháng {self.expand_month(p2)}"

            return self._sequence_expander.expand_sequence(month_str)

    class SequenceExpander:
        """Handles expansion of sequences."""

        def __init__(self, number_expander):
            self._number_expander = number_expander

        def expand_sequence(self, sequence: str, english: bool = False) -> str:
            """Expand sequence patterns."""
            sequence = sequence.lower().replace(".", "")

            # u.23, u23 - I am u30 now :(
            if re.match(r"^u\.?\d{2}$", sequence):
                return f"u {self._number_expander.expand_number(sequence[1:])}"

            result = []
            for char in sequence:
                if char in (
                    mapping := (
                        SEQUENCE_PHONES_EN_MAPPING
                        if english
                        else SEQUENCE_PHONES_VI_MAPPING
                    )
                ):
                    result.append(mapping[char])
                elif char.isnumeric():
                    result.append(NUMBER_MAPPING[char])
                elif char in MONEY_UNITS_MAPPING:
                    result.append(MONEY_UNITS_MAPPING[char])
                else:
                    result.append(char)

            return " ".join(result)

    class AbbreviationExpander:
        """Handles expansion of abbreviations using language models."""

        def __init__(
            self, abbr_dict: Dict[str, List[str]], number_expander, sequence_expander
        ):
            self._abbr_dict = abbr_dict
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander
            self._window_size = 8

            model_path = get_model_weights_path() / "abbreviation_expander"
            session_options = SessionOptions()
            session_options.graph_optimization_level = (
                GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._abbr_model = InferenceSession(
                model_path / "bert.opt.infer.quant.onnx",
                session_options,
                providers=["CPUExecutionProvider"],
            )
            self._abbr_tokenizer = Tokenizer.from_file(
                str(model_path / "tokenizer.json")
            )
            self._abbr_tokenizer.enable_padding(
                pad_id=self._abbr_tokenizer.token_to_id("<pad>"), length=32
            )
            self._abbr_tokenizer.enable_truncation(max_length=32)

        def expand_abbreviation(
            self, abbr: str, left_context: str, right_context: str
        ) -> str:
            """Expand an abbreviation using context."""
            abbr = re.sub(r"\s*-\s*", "", abbr)
            parts = list(filter(len, re.split(r"\.+|\s+", abbr)))

            if "".join(parts) in self._abbr_dict and len(parts) > 1:
                return self.expand_abbreviation(
                    "".join(parts), left_context, right_context
                )

            if len(parts) > 1:
                result = []
                for i, part in enumerate(parts):
                    if part.isnumeric():
                        result.append(self._number_expander.expand_number(part))
                    else:
                        result.append(
                            self.expand_abbreviation(
                                part.replace(".", ""),
                                self._get_left_context(
                                    " ".join([left_context, " ".join(result)])
                                ),
                                self._get_right_context(
                                    " ".join([" ".join(parts[i + 1 :]), right_context])
                                ),
                            )
                        )
                return " ".join(result)
            elif len(parts):
                abbr = parts[0]
            else:
                return ""

            # Handle patterns like "ABC123"
            if re.match(r"^[^0-9]+\d+$", abbr):
                text, number = re.search(r"^([^0-9]+)(\d+)$", abbr).groups()[:2]
                return (
                    self.expand_abbreviation(text, left_context, right_context)
                    + " "
                    + self._number_expander.expand_number(number)
                )

            if abbr in self._abbr_dict:
                if len(self._abbr_dict[abbr]) == 1:
                    return self._abbr_dict[abbr][0]
                else:
                    # Use language model to choose best expansion
                    candidates = [
                        (
                            self._calculate_perplexity(
                                " ".join([left_context, candidate, right_context])
                            ),
                            candidate,
                        )
                        for candidate in self._abbr_dict[abbr]
                    ]
                    candidates.sort()
                    return candidates[0][1]

            return self._sequence_expander.expand_sequence(abbr)

        def _prepare_abbr_input(self, sentence: str) -> Tuple[np.ndarray, np.ndarray]:
            """Prepare input for the abbreviation language model."""
            input_ids = np.array(self._abbr_tokenizer.encode(sentence).ids)[
                np.newaxis, ...
            ]
            seq_len = input_ids.shape[1]
            repeat_input = np.tile(input_ids, (seq_len - 2, 1)).astype(dtype=np.int64)

            mask = np.eye(seq_len, seq_len, k=1, dtype=np.int64)[:-2]
            mask[repeat_input == 1] = 0
            mask_token_id = self._abbr_tokenizer.token_to_id("<mask>")
            masked_input = np.where(mask == 1, mask_token_id, repeat_input)
            labels = np.where(masked_input != mask_token_id, -100, repeat_input)

            return masked_input, labels

        def _calculate_perplexity(self, sentence: str, lower: bool = True) -> float:
            """Calculate perplexity of a sentence using the language model."""
            sentence = sentence.lower() if lower else sentence
            masked_input, labels = self._prepare_abbr_input(sentence)

            ort_inputs = {self._abbr_model.get_inputs()[0].name: masked_input}
            logits = self._abbr_model.run(None, ort_inputs)[0]

            # Calculate cross-entropy loss
            logits_flat = logits.reshape(-1, logits.shape[-1])
            labels_flat = labels.reshape(-1)

            valid_mask = labels_flat != -100
            logits_valid = logits_flat[valid_mask]
            labels_valid = labels_flat[valid_mask]

            log_probs = self._log_softmax(logits_valid)
            loss = -np.mean(log_probs[np.arange(len(labels_valid)), labels_valid])

            return float(loss)

        def _log_softmax(self, x: np.ndarray) -> np.ndarray:
            """Compute log softmax using numpy."""
            x_max = np.max(x, axis=-1, keepdims=True)
            x_stable = x - x_max
            exp_x = np.exp(x_stable)
            sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
            softmax = exp_x / sum_exp_x
            return np.log(softmax + 1e-8)

        def _get_left_context(self, text: str) -> str:
            """Get left context window."""
            return " ".join(text.split()[-self._window_size :])

        def _get_right_context(self, text: str) -> str:
            """Get right context window."""
            return " ".join(text.split()[: self._window_size])

    class ForeignWordExpander:
        """Handles expansion of foreign words."""

        def __init__(self, vn_dict: Set[str], number_expander, sequence_expander):
            self._vn_dict = vn_dict
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander

        def expand_foreign_word(self, word: str) -> str:
            """Expand foreign word."""
            return " ".join(
                self._expand_one_word(w) for w in re.split(r"-+|\s+|\.+|_+", word)
            )

        def _expand_one_word(self, word: str) -> str:
            """Expand numeric word or return unchanged. Need more refined handling later."""
            if word.isnumeric():
                return self._number_expander.expand_number(word)
            return word

    class QuarterExpander:
        """Handles expansion of quarter notation."""

        def __init__(self, number_expander):
            self._number_expander = number_expander

        def expand_quarter(self, quarter: str) -> str:
            """Expand quarter notation (e.g., I/2023)."""
            roman, year = quarter.split("/")[:2]
            return (
                f"{self._number_expander.expand_roma(roman)} "
                f"năm {self._number_expander.expand_number(year)}"
            )

    class VersionExpander:
        """Handles expansion of version numbers."""

        def __init__(self, number_expander, sequence_expander):
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander

        def expand_version(self, version: str) -> str:
            """Expand version numbers."""
            if re.match(r"^\d(\.\d+)*$", version):
                return " chấm ".join(
                    self._number_expander.expand_number(num)
                    for num in version.split(".")
                )
            return self._sequence_expander.expand_sequence(version)

    class FractionExpander:
        """Handles expansion of fractions."""

        def __init__(self, number_expander):
            self._number_expander = number_expander

        def expand_fraction(self, fraction: str) -> str:
            """Expand fractions."""
            parts = list(filter(len, re.split(r"[/:]", fraction)))
            return " trên ".join(
                self._number_expander.expand_number(part) for part in parts
            )

    class MoneyExpander:
        """Handles expansion of money amounts."""

        def __init__(self, number_expander):
            self._number_expander = number_expander

        def expand_money(self, money: str) -> str:
            """Expand money amounts."""
            value = re.sub(r"[^0-9,.\s]", "", money)
            unit = re.sub(r"[0-9,.\s]", "", money)
            result = self._number_expander.expand_number(value)
            if unit in MONEY_UNITS_MAPPING:
                result += f" {MONEY_UNITS_MAPPING[unit]}"
            return result.strip()

    class ScoreExpander:
        """Handles expansion of scores."""

        def __init__(self, number_expander, sequence_expander):
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander

        def expand_score(self, score: str) -> str:
            """Expand scores."""
            if re.match(r"[0-9.,]+([-:/][0-9.,]+)+", score):
                return " ".join(
                    self._number_expander.expand_number(x)
                    for x in re.findall(r"[0-9.,]+", score)
                )
            return self._sequence_expander.expand_sequence(score)

    class RangeExpander:
        """Handles expansion of ranges."""

        def __init__(self, number_expander, sequence_expander):
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander

        def expand_range(self, range_str: str) -> str:
            """Expand ranges."""
            if re.match(r"[0-9.,]+([-:][0-9.,]+)+", range_str):
                return " đến ".join(
                    self._number_expander.expand_number(x)
                    for x in re.findall(r"[0-9.,]+", range_str)
                )
            return self._sequence_expander.expand_sequence(range_str)

    class PercentExpander:
        """Handles expansion of percentages."""

        def __init__(self, number_expander, sequence_expander):
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander

        def expand_percent(self, percent: str) -> str:
            """Expand percentages."""
            percent = percent.strip()

            # number %
            if re.match(r"^-?[0-9.,]+%$", percent):
                return f"{self._number_expander.expand_number(percent[:-1])} phần trăm"
            # number - number %
            elif re.match(r"^[0-9.,]+\s*-\s*[0-9.,]+%$", percent):
                p1, p2 = re.split(r"\s*-\s*", percent)[:2]
                return f"{self._number_expander.expand_number(p1)} đến {self.expand_percent(p2)}"
            # number % - number %
            elif re.match(r"^[0-9.,]+%\s*-\s*[0-9.,]+%$", percent):
                p1, p2 = re.split(r"\s*-\s*", percent)[:2]
                return f"{self.expand_percent(p1)} đến {self.expand_percent(p2)}"

            return self._sequence_expander.expand_sequence(percent)

    class MeasureExpander:
        """Handles expansion of measurements."""

        def __init__(self, number_expander, sequence_expander, vn_dict):
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander
            self._vn_dict = vn_dict

        def expand_measure(self, measure: str) -> str:
            """Expand measurements."""
            measure = measure.strip()

            # Pure number
            if re.match(r"^[0-9.,]+$", measure):
                return self._number_expander.expand_number(measure)
            # Units only
            elif re.match(r"^[^0-9/][^/]*(\s*/\s*[^/]+)*$", measure):
                units = re.split(r"\s*/\s*", measure)
                return " trên ".join(self._expand_unit(unit) for unit in units)
            # number - number unit
            elif re.match(r"^[0-9.,]+\s*-\s*.*$", measure):
                n, m = re.split(r"\s*-\s*", measure)[:2]
                return f"{self._number_expander.expand_number(n)} đến {self.expand_measure(m)}"
            # number unit
            elif re.match(r"^[0-9.,]+[^0-9.,][^.,]*$", measure):
                n, u = re.search(r"^([0-9.,]+)([^0-9.,][^.,]*)$", measure).groups()[:2]
                return (
                    f"{self._number_expander.expand_number(n)} {self.expand_measure(u)}"
                )

            return self._expand_unit(measure)

        def _expand_unit(self, unit: str) -> str:
            """Expand a unit."""
            if unit in MEASUREMENT_UNITS_MAPPING:
                return MEASUREMENT_UNITS_MAPPING[unit]
            elif unit in MONEY_UNITS_MAPPING:
                return MONEY_UNITS_MAPPING[unit]
            elif unit in self._vn_dict:
                return unit
            return self._sequence_expander.expand_sequence(unit)

    class UrlExpander:
        """Handles expansion of URLs using lexicon-based maximum matching (https://arxiv.org/abs/2209.02971)."""

        def __init__(self, sequence_expander, nosign_dict):
            self._sequence_expander = sequence_expander
            self._nosign_dict = nosign_dict

        def expand_url(self, url: str) -> str:
            """Expand URLs using lexicon-based maximum matching."""
            url = url.strip()
            result = []
            tokens = list(url)
            min_window = 1
            max_window = len(tokens)
            start_idx = 0

            while start_idx <= len(tokens) - min_window:
                found = False

                for window_size in range(max_window, min_window - 1, -1):
                    end_idx = start_idx + window_size - 1
                    candidate = tokens[start_idx : end_idx + 1]
                    candidate_str = "".join(candidate)

                    if (
                        candidate_str.lower() in self._nosign_dict
                        and len(candidate_str) > 1
                    ):
                        result.append(candidate_str)
                        start_idx = end_idx + 1
                        found = True
                        break

                if not found:
                    result.append(
                        self._sequence_expander.expand_sequence(tokens[start_idx])
                    )
                    start_idx += 1

            if start_idx < len(tokens):
                result.append(
                    self._sequence_expander.expand_sequence(tokens[start_idx])
                )

            return " ".join(result)

    def __init__(
        self,
        vn_dict: Union[List[str], None] = None,
        abbr_dict: Union[Dict[str, List[str]], None] = None,
    ):
        """Initialize the rule-based NSW expander.
        Args:
            vn_dict: A list of Vietnamese words. If None, the default Vietnamese dictionary will be used.
            abbr_dict: A dictionary of abbreviations and their expansions. If None, the default abbreviation dictionary will be used.
        """
        self._vn_dict = set(vn_dict or load_vietnamese_syllables())
        self._abbr_dict = abbr_dict or load_abbreviation_dict()
        self._nosign_dict = set(map(unidecode, self._vn_dict))
        self._no_norm_list = [".", ",", "...", "-"]

        self._number_expander = self.NumberExpander()
        self._sequence_expander = self.SequenceExpander(self._number_expander)
        self._time_date_expander = self.TimeDateExpander(
            self._number_expander, self._sequence_expander
        )
        self._abbr_expander = self.AbbreviationExpander(
            self._abbr_dict, self._number_expander, self._sequence_expander
        )
        self._foreign_expander = self.ForeignWordExpander(
            self._vn_dict, self._number_expander, self._sequence_expander
        )
        self._quarter_expander = self.QuarterExpander(self._number_expander)
        self._version_expander = self.VersionExpander(
            self._number_expander, self._sequence_expander
        )
        self._fraction_expander = self.FractionExpander(self._number_expander)
        self._money_expander = self.MoneyExpander(self._number_expander)
        self._score_expander = self.ScoreExpander(
            self._number_expander, self._sequence_expander
        )
        self._range_expander = self.RangeExpander(
            self._number_expander, self._sequence_expander
        )
        self._percent_expander = self.PercentExpander(
            self._number_expander, self._sequence_expander
        )
        self._measure_expander = self.MeasureExpander(
            self._number_expander, self._sequence_expander, self._vn_dict
        )
        self._url_expander = self.UrlExpander(
            self._sequence_expander, self._nosign_dict
        )

        self._expanders = {
            "LSEQ": self._sequence_expander.expand_sequence,
            "MEA": self._measure_expander.expand_measure,
            "MONEY": self._money_expander.expand_money,
            "NDAT": self._time_date_expander.expand_date,
            "NDAY": self._time_date_expander.expand_day,
            "NDIG": self._number_expander.expand_digit,
            "NFRC": self._fraction_expander.expand_fraction,
            "NMON": self._time_date_expander.expand_month,
            "NNUM": self._number_expander.expand_number,
            "NPER": self._percent_expander.expand_percent,
            "NQUA": self._quarter_expander.expand_quarter,
            "NRNG": self._range_expander.expand_range,
            "NSCR": self._score_expander.expand_score,
            "NTIM": self._time_date_expander.expand_time,
            "NVER": self._version_expander.expand_version,
            "ROMA": self._number_expander.expand_roma,
            "URLE": self._url_expander.expand_url,
            "LWRD": self._foreign_expander.expand_foreign_word,
        }

    def expand(self, words: List[str], tags: List[str]) -> List[str]:
        """Expand a list of words based on their tags."""
        if len(words) != len(tags):
            raise ValueError("The length of words and corresponding tags do not match!")

        results = []
        current_group = []

        i = 0
        while i < len(words):
            if tags[i] == "O":
                # Normal words
                word = words[i]
                if (
                    word.lower() in self._vn_dict and len(word) > 1
                ) or word in self._no_norm_list:
                    results.append(word)
                else:
                    results.extend(
                        self._sequence_expander.expand_sequence(part)
                        for part in re.split(r"[/\\]", word)
                    )
                i += 1
            else:
                # Non-standard words
                current_group.append(words[i])
                tag_type = tags[i][2:]  # Remove B- or I- prefix
                i += 1

                # Collect all words in the same group
                while i < len(words) and not (
                    tags[i] == "O"
                    or tags[i][2:] != tag_type
                    or tags[i].startswith("B-")
                ):
                    current_group.append(words[i])
                    i += 1

                # Expand the group
                if tag_type in self._expanders:
                    results.append(self._expanders[tag_type](" ".join(current_group)))
                elif tag_type == "LABB":
                    results.append(
                        self._abbr_expander.expand_abbreviation(
                            " ".join(current_group),
                            " ".join(results),
                            " ".join(words[i:]),
                        )
                    )
                else:
                    results.append(" ".join(current_group))

                current_group = []

        return results
