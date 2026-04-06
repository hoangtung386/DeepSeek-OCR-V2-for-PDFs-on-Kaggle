"""Tests for markdown output writing."""

from src.postprocessing.markdown_writer import write_markdown


class TestWriteMarkdown:
    def test_single_page(self, tmp_path):
        output_file = tmp_path / "result.md"
        write_markdown(["Hello world"], str(output_file))

        content = output_file.read_text(encoding="utf-8")
        assert "<!-- Page 1 -->" in content
        assert "Hello world" in content

    def test_multiple_pages(self, tmp_path):
        output_file = tmp_path / "result.md"
        pages = ["Page one text", "Page two text", "Page three text"]
        write_markdown(pages, str(output_file))

        content = output_file.read_text(encoding="utf-8")
        assert "<!-- Page 1 -->" in content
        assert "<!-- Page 2 -->" in content
        assert "<!-- Page 3 -->" in content
        assert content.count("---") == 2
        assert "Page one text" in content
        assert "Page three text" in content

    def test_creates_parent_directories(self, tmp_path):
        output_file = tmp_path / "nested" / "dir" / "result.md"
        write_markdown(["test"], str(output_file))

        assert output_file.exists()
        assert "test" in output_file.read_text(encoding="utf-8")

    def test_empty_pages(self, tmp_path):
        output_file = tmp_path / "empty.md"
        write_markdown([], str(output_file))

        content = output_file.read_text(encoding="utf-8")
        assert content == ""
