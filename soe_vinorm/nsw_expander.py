import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set, Union

import numpy as np
from onnxruntime import InferenceSession
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

    @staticmethod
    def _identity(text: str) -> str:
        """Identity function that returns input unchanged."""
        return text

    class NumberExpander:
        """
        Handle expansion of numbers, digits, and numeric patterns.
        """

        def __init__(self):
            self._op_map = str.maketrans(
                {"+": "cộng", "-": "trừ", "*": "nhân", "/": "chia", "^": "mũ"}
            )

        def expand_digit(self, digit: str) -> str:
            """Expand a single digit to its spoken form."""
            result = []
            for char in digit.replace(" ", ""):
                if char in NUMBER_MAPPING:
                    result.append(NUMBER_MAPPING[char])
                else:
                    result.append(char)

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
                while len(number) > 1 and number[0] == "0" and number[1].isdigit():
                    number = number[1:]

                if len(re.findall(r"[-+]?[0-9.,]+", number)) > 1:
                    return (
                        re.sub(
                            r"\s*([-+]?[0-9.,]+)\s*",
                            lambda m: f" {self.expand_number(m.group(1))} ",
                            number,
                        )
                        .strip()
                        .translate(self._op_map)
                    )

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

                return f"{sign} {result} {decimal_part}".strip()

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
        """Handle expansion of time and date patterns."""

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
            # HH[hg](MM)* [-] HH[hg](MM)*
            elif re.match(r"^\d{1,2}[hg]\d{0,2}\s*-\s*\d{1,2}[hg]\d{0,2}$", time_str):
                h1, h2 = re.split(r"\s*-\s*", time_str)[:2]
                return f"{self.expand_time(h1)} đến {self.expand_time(h2)}"
            # HH[:]MM [-] HH[:]MM
            elif re.match(r"^\d{1,2}:\d{1,2}\s*-\s*\d{1,2}:\d{1,2}$", time_str):
                t1, t2 = re.split(r"\s*-\s*", time_str)[:2]
                return f"{self.expand_time(t1)} đến {self.expand_time(t2)}"

            return self._sequence_expander.expand_sequence(time_str)

        def expand_day(self, day_str: str) -> str:
            """Expand day patterns to spoken form."""
            day_str = day_str.strip().replace(".", "/")

            # DD[/-]MM
            if re.match(r"^\d{1,2}\s*[/-]\s*\d{1,2}$", day_str):
                d, m = re.split(r"\s*/\s*|\s*-\s*", day_str)[:2]
                return (
                    f"{self._number_expander.expand_number(d)} tháng "
                    f"{self._number_expander.expand_number(m)}"
                )
            # DD [-] DD[/]MM
            elif re.match(r"^\d{1,2}\s*-\s*\d{1,2}/\d{1,2}$", day_str):
                d1, d2 = re.split(r"\s*-\s*", day_str)[:2]
                return (
                    f"{self._number_expander.expand_number(d1)} đến ngày "
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
                return (
                    f"{self._number_expander.expand_number(d)} tháng "
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
                return (
                    f"{self._number_expander.expand_number(d1)} đến ngày "
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
        """Handle expansion of sequences."""

        def __init__(self, number_expander, vn_dict: Set[str], no_norm_list: List[str]):
            self._number_expander = number_expander
            self._vn_dict = vn_dict
            self._no_norm_list = no_norm_list

        def expand_sequence(self, sequence: str, english: bool = False) -> str:
            """Expand sequence patterns."""
            sequence_lower = sequence.lower()

            # u.23, u23 - I am u30 now :(
            if re.match(r"^u\.?\d{2}$", sequence_lower):
                return f"u {self._number_expander.expand_number(sequence_lower[1:])}"

            # Only expand single character words if in english mode
            if sequence_lower in self._vn_dict and (
                not english or len(sequence_lower) > 1
            ):
                return sequence

            result = []
            idx = 0
            while idx < len(sequence_lower):
                char = sequence_lower[idx]
                if not char.strip():
                    idx += 1
                    continue

                if char in self._no_norm_list and not english:
                    result.append(char)
                    idx += 1
                    continue

                if char in (
                    mapping := (
                        SEQUENCE_PHONES_EN_MAPPING
                        if english
                        else SEQUENCE_PHONES_VI_MAPPING
                    )
                ):
                    result.append(
                        mapping[char].capitalize()
                        if sequence[idx].isupper()
                        else mapping[char]
                    )
                    idx += 1
                elif char.isdigit():
                    digit_idx = idx
                    number = ""
                    while (
                        digit_idx < len(sequence_lower)
                        and sequence_lower[digit_idx].isdigit()
                    ):
                        number += sequence_lower[digit_idx]
                        digit_idx += 1

                    if number[0] != "0" and len(number) <= 4:
                        result.append(self._number_expander.expand_number(number))
                    else:
                        result.extend(NUMBER_MAPPING[c] for c in number)
                    idx = digit_idx
                elif char in MONEY_UNITS_MAPPING:
                    result.append(MONEY_UNITS_MAPPING[char])
                    idx += 1
                else:
                    result.append(char)
                    idx += 1

            return " ".join(result)

    class AbbreviationExpander:
        """Handle expansion of abbreviations using a likelihood scorer."""

        def __init__(
            self,
            number_expander,
            sequence_expander,
            abbr_dict: Dict[str, List[str]],
            model_path: Union[str, None] = None,
        ):
            self._abbr_dict = abbr_dict
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander

            if model_path is None:
                model_path = get_model_weights_path()
            else:
                model_path = Path(model_path)

            model_path = model_path / "abbreviation_expander" / "v0.2"

            self._scorer = InferenceSession(
                model_path / "scorer.onnx",
                providers=["CPUExecutionProvider"],
            )

            with open(model_path / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)

            self._window_size = config["window_size"]
            self._vocab = config["vocab"]
            self._seq_len = config["seq_len"]

        def expand_abbreviation(
            self,
            abbr: str,
            left_context: str,
            right_context: str,
            attempt_cleaning: bool = True,
        ) -> str:
            """Expand an abbreviation using context."""

            # Handle patterns like "ABC123"
            if match := re.match(r"^([^0-9]+)(\d+)$", abbr):
                text, number = match.groups()
                return (
                    f"{self.expand_abbreviation(text, left_context, right_context, attempt_cleaning)} "
                    f"{self._number_expander.expand_number(number)}"
                )

            if abbr in self._abbr_dict:
                if len(self._abbr_dict[abbr]) == 1:
                    return self._abbr_dict[abbr][0]
                else:
                    # Use a likelihood scorer to choose the best expansion
                    input_ids = self._prepare_input(
                        self._abbr_dict[abbr], left_context, right_context
                    )
                    scores = self._scorer.run(
                        None, {self._scorer.get_inputs()[0].name: input_ids}
                    )[0]
                    return self._abbr_dict[abbr][np.argmax(scores)]
            elif attempt_cleaning:
                # Remove hyphens
                abbr = re.sub(r"\s*-\s*", "", abbr)
                # Split the abbreviation into parts
                parts = list(filter(None, re.split(r"\.+|\s+|(\d+)", abbr)))

                if not parts:
                    return ""

                # If the joined abbreviation is in the dictionary, expand it recursively
                if (new_abbr := "".join(parts)) in self._abbr_dict:
                    return self.expand_abbreviation(
                        new_abbr, left_context, right_context, False
                    )

                # Otherwise, expand parts one by one
                result = []
                for part in parts:
                    if part.isdigit():
                        result.append(self._number_expander.expand_number(part))
                    else:
                        result.append(
                            self.expand_abbreviation(
                                part,
                                " ".join([left_context, *result]),
                                f"LABB {right_context}",
                                False,
                            )
                        )
                return " ".join(result)

            return self._sequence_expander.expand_sequence(abbr)

        def _prepare_input(
            self, candidates: List[str], left_context: str, right_context: str
        ) -> np.ndarray:
            """Prepare input for likelihood scorer."""
            left_context = self._get_left_context(left_context)
            right_context = self._get_right_context(right_context)
            examples = [
                f"{left_context} {candidate.lower()} {right_context}".strip()
                for candidate in candidates
            ]

            input_ids = []
            for example in examples:
                ids = [
                    self._vocab[token] if token in self._vocab else self._vocab["<unk>"]
                    for token in example.split()
                ]
                if len(ids) < self._seq_len:
                    ids += [self._vocab["<pad>"]] * (self._seq_len - len(ids))
                input_ids.append(ids[: self._seq_len])

            return np.array(input_ids, dtype=np.int64)

        def _get_left_context(self, text: str) -> str:
            """Get left context window."""
            return " ".join(text.split()[-self._window_size :])

        def _get_right_context(self, text: str) -> str:
            """Get right context window."""
            return " ".join(text.split()[: self._window_size])

    class ForeignWordExpander:
        """Handle expansion of foreign words."""

        def __init__(self, number_expander, sequence_expander, vn_dict: Set[str]):
            self._vn_dict = vn_dict
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander

        def expand_foreign_word(self, word: str) -> str:
            """Expand foreign word."""
            return " ".join(
                self._expand_one_word(w) for w in re.split(r"-+|\s+|\.+|_+", word) if w
            )

        def _expand_one_word(self, word: str) -> str:
            """Expand numeric word or return unchanged. Need more refined handling later."""
            if word.isdigit():
                return self._number_expander.expand_number(word)
            return word

    class QuarterExpander:
        """Handle expansion of quarter notation."""

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
        """Handle expansion of version numbers."""

        def __init__(self, number_expander, sequence_expander):
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander

        def expand_version(self, version: str) -> str:
            """Expand version numbers."""
            if re.match(r"^\d+(\.\d+)*$", version):
                return " chấm ".join(
                    self._number_expander.expand_number(num)
                    for num in version.split(".")
                )
            return self._sequence_expander.expand_sequence(version)

    class FractionExpander:
        """Handle expansion of fractions."""

        def __init__(self, number_expander):
            self._number_expander = number_expander

        def expand_fraction(self, fraction: str) -> str:
            """Expand fractions."""
            parts = list(filter(len, re.split(r"[/:]", fraction)))
            return " trên ".join(
                self._number_expander.expand_number(part) for part in parts if part
            )

    class MoneyExpander:
        """Handle expansion of money amounts."""

        def __init__(self, number_expander):
            self._number_expander = number_expander

        def expand_money(self, money: str) -> str:
            """Expand money amounts."""
            value = re.sub(r"[^0-9,.\s]", "", money)
            unit = re.sub(r"[0-9,.\s]", "", money)

            result = self._number_expander.expand_number(value)
            if unit in MONEY_UNITS_MAPPING:
                result += f" {MONEY_UNITS_MAPPING[unit]}"
            else:
                result += f" {unit}"

            return result.strip()

    class ScoreExpander:
        """Handle expansion of scores."""

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
        """Handle expansion of ranges."""

        def __init__(self, number_expander, sequence_expander):
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander

        def expand_range(self, range_str: str) -> str:
            """Expand ranges."""
            if re.match(r"[0-9.,]+([-–:][0-9.,]+)+", range_str):
                return " đến ".join(
                    self._number_expander.expand_number(x)
                    for x in re.findall(r"[0-9.,]+", range_str)
                )
            return self._sequence_expander.expand_sequence(range_str)

    class PercentExpander:
        """Handle expansion of percentages."""

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
        """Handle expansion of measurements."""

        def __init__(self, number_expander, sequence_expander, vn_dict):
            self._number_expander = number_expander
            self._sequence_expander = sequence_expander
            self._vn_dict = vn_dict

        def expand_measure(self, measure: str) -> str:
            """Expand measurements."""
            measure = measure.strip()

            # Pure number
            if re.match(r"^-?[0-9.,]+$", measure):
                return self._number_expander.expand_number(measure)
            # Units only
            elif re.match(r"^[^0-9/][^/]*(\s*/\s*[^/]+)*$", measure):
                units = re.split(r"\s*/\s*", measure)
                return " trên ".join(self._expand_unit(unit) for unit in units)
            # number - number unit
            elif re.match(r"^-?[0-9.,]+\s*[-–]\s*.*$", measure):
                n, m = re.split(r"\s*[-–]\s*", measure)[:2]
                return f"{self._number_expander.expand_number(n)} đến {self.expand_measure(m)}"
            # number unit
            elif re.match(r"^-?[0-9.,]+[^0-9.,][^.,]*$", measure):
                n, u = re.search(r"^(-?[0-9.,]+)([^0-9.,][^.,]*)$", measure).groups()[
                    :2
                ]
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
        """Handle expansion of URLs using lexicon-based maximum matching (https://arxiv.org/abs/2209.02971)."""

        def __init__(self, sequence_expander, no_tone_dict):
            self._sequence_expander = sequence_expander
            self._no_tone_dict = no_tone_dict

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
                        candidate_str.lower() in self._no_tone_dict
                        and len(candidate_str) > 1
                    ):
                        result.append(candidate_str)
                        start_idx = end_idx + 1
                        found = True
                        break

                if not found:
                    result.append(
                        self._sequence_expander.expand_sequence(
                            tokens[start_idx],
                            english=tokens[start_idx] not in ["@", "."],
                        )
                    )
                    start_idx += 1

            if start_idx < len(tokens):
                result.append(
                    self._sequence_expander.expand_sequence(
                        tokens[start_idx], english=tokens[start_idx] not in ["@", "."]
                    )
                )

            return re.sub(r"\s*\.\s*", " chấm ", " ".join(filter(len, result))).strip()

    def __init__(
        self,
        model_path: Union[str, None] = None,
        vn_dict: Union[List[str], None] = None,
        abbr_dict: Union[Dict[str, List[str]], None] = None,
        expand_url: bool = True,
        **kwargs,
    ):
        """Initialize the rule-based NSW expander.

        Args:
            model_path: Path to the model repository directory. If None, use default path.
            vn_dict: List of Vietnamese words for dictionary lookup. If None, use default Vietnamese dictionary.
            abbr_dict: Dictionary of abbreviations and their expansions. If None, use default abbreviation dictionary.
            expand_url: Whether to expand URLs. If True, expand URLs. If False, return URLs unchanged.
        """
        self._vn_dict = set(vn_dict or load_vietnamese_syllables())
        self._abbr_dict = abbr_dict or load_abbreviation_dict()
        self._no_tone_dict = set(map(unidecode, self._vn_dict))

        # no normalization for these characters
        self._no_norm_list = [
            ".",
            ",",
            ":",
            ";",
            "!",
            "?",
            "...",
            "-",
            "–",
            "/",
            "\\",
            "~",
        ]

        self._number_expander = self.NumberExpander()
        self._sequence_expander = self.SequenceExpander(
            self._number_expander, self._vn_dict, self._no_norm_list
        )
        self._time_date_expander = self.TimeDateExpander(
            self._number_expander, self._sequence_expander
        )
        self._abbr_expander = self.AbbreviationExpander(
            self._number_expander,
            self._sequence_expander,
            self._abbr_dict,
            model_path,
        )
        self._foreign_expander = self.ForeignWordExpander(
            self._number_expander, self._sequence_expander, self._vn_dict
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
            self._sequence_expander, self._no_tone_dict
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
            "URLE": self._url_expander.expand_url if expand_url else self._identity,
            "LWRD": self._foreign_expander.expand_foreign_word,
        }

    def expand(self, words: List[str], tags: List[str]) -> List[str]:
        """Expand a list of words based on their tags."""
        if len(words) != len(tags):
            raise ValueError("The length of words and corresponding tags do not match!")

        results = []
        current_group = []

        left_context = []
        right_context = [
            words[i].lower() if tags[i] == "O" else tags[i][2:]
            for i in range(len(words))
        ]

        i = 0
        while i < len(words):
            if tags[i] == "O":
                # Normal words
                word = words[i]
                word_lower = word.lower()
                left_context.append(word_lower)

                # Is in default Vietnamese dictionary or no tone dictionary
                if (
                    (word_lower in self._vn_dict or word_lower in self._no_tone_dict)
                    and len(word) > 1
                    or word in self._no_norm_list
                ):
                    results.append(word)
                # In case the detector does not work well, only consider numbers with 1-8 digits
                elif re.match(r"^-?\d[\d.,]{0,8}$", word):
                    results.append(self._number_expander.expand_number(word))
                else:
                    results.extend(
                        self._sequence_expander.expand_sequence(part)
                        for part in re.split(r"([/\\.])", word)
                        if part
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
                nsw = " ".join(current_group)
                if tag_type in self._expanders:
                    results.append(self._expanders[tag_type](nsw))
                    left_context.append(tag_type)
                elif tag_type == "LABB":
                    expanded_abbreviation = self._abbr_expander.expand_abbreviation(
                        nsw,
                        " ".join(left_context),
                        " ".join(right_context[i:]),
                    )
                    results.append(expanded_abbreviation)
                    left_context.append(expanded_abbreviation.lower())
                else:
                    results.append(nsw)
                    left_context.append(tag_type)

                current_group = []

        return results
