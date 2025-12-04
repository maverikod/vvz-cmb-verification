"""
Unit tests for data_index_loader module - basic functionality.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from utils.io.data_index_loader import DataIndex


class TestDataIndexLoading:
    """Tests for DataIndex.load() method."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML index file."""
        test_data = {
            "data_index": {
                "test_category": {
                    "description": "Test category",
                    "file": "test_file.txt",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)

            assert isinstance(index, DataIndex)
            assert index.index_data == test_data
            assert index.index_path == index_path

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises FileNotFoundError."""
        index_path = Path("/nonexistent/path/data_index.yaml")

        with pytest.raises(FileNotFoundError):
            DataIndex.load(index_path)

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises YAMLError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                f.write("invalid: yaml: content: [unclosed")

            with pytest.raises(yaml.YAMLError):
                DataIndex.load(index_path)

    def test_load_with_default_path(self):
        """Test loading with None path uses config default."""
        # This test requires actual data_index.yaml to exist
        # We'll test that it doesn't crash
        try:
            index = DataIndex.load()
            assert isinstance(index, DataIndex)
        except (FileNotFoundError, yaml.YAMLError):
            # If file doesn't exist or is invalid, that's expected
            # in test environment
            pytest.skip(
                "data_index.yaml not found or invalid in test environment"
            )


class TestGetFilePath:
    """Tests for get_file_path() method."""

    def test_get_file_path_single_file(self):
        """Test getting path for category with single file."""
        test_data = {
            "data_index": {
                "test_category": {
                    "file": "test_file.txt",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)

            # Create the file
            test_file = Path(tmpdir) / "test_file.txt"
            test_file.write_text("test content")

            path = index.get_file_path("test_category", "test_file.txt")
            assert path is not None
            assert path.name == "test_file.txt"

    def test_get_file_path_with_directory(self):
        """Test getting path for file in directory."""
        test_data = {
            "data_index": {
                "test_category": {
                    "directory": "subdir",
                    "files": [{"name": "test_file.txt"}],
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)

            # Create directory and file
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            test_file = subdir / "test_file.txt"
            test_file.write_text("test content")

            path = index.get_file_path("test_category", "test_file.txt")
            assert path is not None
            assert path.name == "test_file.txt"

    def test_get_file_path_nonexistent(self):
        """Test getting path for non-existent file returns None."""
        test_data = {
            "data_index": {
                "test_category": {
                    "file": "test_file.txt",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)

            path = index.get_file_path("test_category", "nonexistent.txt")
            assert path is None

    def test_get_file_path_invalid_category(self):
        """Test getting path for invalid category returns None."""
        test_data = {"data_index": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)

            path = index.get_file_path("invalid_category", "file.txt")
            assert path is None


class TestGetFilesByCategory:
    """Tests for get_files_by_category() method."""

    def test_get_files_single_file(self):
        """Test getting files for category with single file."""
        test_data = {
            "data_index": {
                "test_category": {
                    "description": "Test category",
                    "file": "test_file.txt",
                    "size": "1 KB",
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)
            files = index.get_files_by_category("test_category")

            assert len(files) == 1
            assert files[0]["name"] == "test_file.txt"
            assert files[0]["category"] == "test_category"
            assert files[0]["size"] == "1 KB"

    def test_get_files_list(self):
        """Test getting files for category with files list."""
        test_data = {
            "data_index": {
                "test_category": {
                    "files": [
                        {"name": "file1.txt", "size": "1 KB"},
                        {"name": "file2.txt", "size": "2 KB"},
                    ]
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)
            files = index.get_files_by_category("test_category")

            assert len(files) == 2
            assert files[0]["name"] == "file1.txt"
            assert files[1]["name"] == "file2.txt"

    def test_get_files_invalid_category(self):
        """Test getting files for invalid category returns empty list."""
        test_data = {"data_index": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "data_index.yaml"
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.dump(test_data, f)

            index = DataIndex.load(index_path)
            files = index.get_files_by_category("invalid_category")

            assert files == []
