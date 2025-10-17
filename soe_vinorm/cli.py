import argparse
import sys
from pathlib import Path
from textwrap import dedent
from typing import List, Union

from soe_vinorm import SoeNormalizer


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="soe-vinorm",
        description="Vietnamese text normalization toolkit - Convert text to spoken form",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
            Examples:
              # Normalize text from stdin
              echo "Năm 2021" | soe-vinorm

              # Normalize texts from a file line by line
              soe-vinorm -i input.txt -o output.txt

              # Process with custom options
              soe-vinorm -i input.txt --no-expand-sequence --no-expand-urle

              # Batch process with parallel workers
              soe-vinorm -i input.txt -o output.txt --n-jobs 4 --show-progress
            """),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input file path. If not specified, read from stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path. If not specified, write to stdout.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model repository directory for loading pre-downloaded weights.",
    )
    parser.add_argument(
        "--no-expand-sequence",
        action="store_false",
        dest="expand_sequence",
        default=True,
        help="Disable expansion of unknown sequences (default: expand enabled).",
    )
    parser.add_argument(
        "--no-expand-urle",
        action="store_false",
        dest="expand_urle",
        default=True,
        help="Disable expansion of URLs and emails (default: expand enabled).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for batch processing (default: 1).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bar during batch processing.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show version information.",
    )

    return parser


def read_input(input_path: Union[str, None] = None) -> List[str]:
    """Read input text from file or stdin."""
    if input_path:
        path = Path(input_path)
        if not path.exists():
            print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]
    else:
        return [line.rstrip("\n") for line in sys.stdin]


def write_output(lines: List[str], output_path: Union[str, None] = None):
    """Write output text to file or stdout."""
    if output_path:
        path = Path(output_path)
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    else:
        for line in lines:
            print(line)


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.version:
        from soe_vinorm import __version__

        print(f"Soe Vinorm {__version__}")
        sys.exit(0)

    try:
        lines = read_input(args.input)
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        kwargs = {
            "expand_sequence": args.expand_sequence,
            "expand_urle": args.expand_urle,
        }
        if args.model_path:
            kwargs["model_path"] = args.model_path

        normalizer = SoeNormalizer(**kwargs)
    except Exception as e:
        print(f"Error initializing normalizer: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        normalized_texts = normalizer.batch_normalize(
            lines, n_jobs=args.n_jobs, show_progress=args.show_progress
        )
    except Exception as e:
        print(f"Error during normalization: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        write_output(normalized_texts, args.output)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
