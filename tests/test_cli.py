from soe_vinorm.cli import create_parser, read_input, write_output


class TestCLIParser:
    """Test the CLI argument parser."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "soe-vinorm"

    def test_parser_default_args(self):
        """Test parser with default arguments."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.input is None
        assert args.output is None
        assert args.expand_sequence is True
        assert args.expand_urle is True
        assert args.n_jobs == 1
        assert args.show_progress is False

    def test_parser_with_input_output(self):
        """Test parser with input and output files."""
        parser = create_parser()
        args = parser.parse_args(["-i", "input.txt", "-o", "output.txt"])
        assert args.input == "input.txt"
        assert args.output == "output.txt"

    def test_parser_disable_options(self):
        """Test parser with disabled options."""
        parser = create_parser()
        args = parser.parse_args(["--no-expand-sequence", "--no-expand-urle"])
        assert args.expand_sequence is False
        assert args.expand_urle is False

    def test_parser_n_jobs(self):
        """Test parser with n_jobs option."""
        parser = create_parser()
        args = parser.parse_args(["--n-jobs", "4"])
        assert args.n_jobs == 4

    def test_parser_show_progress(self):
        """Test parser with show-progress option."""
        parser = create_parser()
        args = parser.parse_args(["--show-progress"])
        assert args.show_progress is True


class TestIOFunctions:
    """Test input/output functions."""

    def test_read_input_from_file(self, tmp_path):
        """Test reading input from a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")

        lines = read_input(str(test_file))
        assert lines == ["Line 1", "Line 2", "Line 3"]

    def test_write_output_to_file(self, tmp_path):
        """Test writing output to a file."""
        output_file = tmp_path / "output.txt"
        lines = ["Line 1", "Line 2", "Line 3"]

        write_output(lines, str(output_file))

        content = output_file.read_text(encoding="utf-8")
        assert content == "Line 1\nLine 2\nLine 3\n"

    def test_write_output_to_stdout(self, capsys):
        """Test writing output to stdout."""
        lines = ["Line 1", "Line 2"]
        write_output(lines, None)

        captured = capsys.readouterr()
        assert captured.out == "Line 1\nLine 2\n"
