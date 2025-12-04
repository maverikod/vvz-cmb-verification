"""
Data index loader and query interface.

Provides functionality to load and query the data index YAML file
for discovering and accessing input data files.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import fnmatch
from config.settings import Config


class DataIndex:
    """
    Data index manager for querying input data files.

    Provides methods to search and access data files based on
    the comprehensive index in data/in/data_index.yaml.
    """

    def __init__(self, index_data: Dict[str, Any], index_path: Path):
        """
        Initialize data index.

        Args:
            index_data: Parsed YAML data from index file
            index_path: Path to index file (for resolving relative paths)
        """
        self.index_data = index_data
        self.index_path = index_path
        self._data_index_root = index_path.parent

    @classmethod
    def load(cls, index_path: Optional[Path] = None) -> "DataIndex":
        """
        Load data index from YAML file.

        Args:
            index_path: Path to data_index.yaml. If None, uses config path.

        Returns:
            DataIndex instance

        Raises:
            FileNotFoundError: If index file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if index_path is None:
            config = Config._load_defaults()
            index_path = config.paths.data_index

        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Data index file not found: {index_path}")

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in data index file: {e}") from e

        if not isinstance(index_data, dict):
            raise ValueError("Data index file must contain a dictionary at root level")

        return cls(index_data, index_path)

    def get_file_path(self, category: str, file_name: str) -> Optional[Path]:
        """
        Get file path for specific data file.

        Args:
            category: Data category (e.g., 'act_dr6_02', 'cmb_microcells')
            file_name: Name of the file

        Returns:
            Path to file, or None if not found
        """
        data_index = self.index_data.get("data_index", {})
        category_data = data_index.get(category)

        if category_data is None:
            return None

        # Check if category has a single file
        if "file" in category_data:
            if category_data["file"] == file_name:
                return self._resolve_path(category_data["file"])

        # Check if category has a directory
        if "directory" in category_data:
            directory = category_data["directory"]
            file_path = self._data_index_root / directory / file_name
            if file_path.exists():
                return file_path

        # Check files list
        if "files" in category_data:
            for file_info in category_data["files"]:
                if isinstance(file_info, dict):
                    if file_info.get("name") == file_name:
                        # Resolve path based on category structure
                        if "directory" in category_data:
                            return self._resolve_path(
                                category_data["directory"], file_name
                            )
                        return self._resolve_path(file_name)
                elif isinstance(file_info, str) and file_info == file_name:
                    return self._resolve_path(file_info)

        return None

    def get_files_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all files in a specific category.

        Args:
            category: Data category name

        Returns:
            List of file information dictionaries
        """
        data_index = self.index_data.get("data_index", {})
        category_data = data_index.get(category)

        if category_data is None:
            return []

        files_list = []

        # Check if category has a single file
        if "file" in category_data:
            file_info = {
                "name": category_data["file"],
                "category": category,
            }
            if "size" in category_data:
                file_info["size"] = category_data["size"]
            if "description" in category_data:
                file_info["description"] = category_data["description"]
            if "content" in category_data:
                file_info["content"] = category_data["content"]
            if "usage" in category_data:
                file_info["usage"] = category_data["usage"]
            files_list.append(file_info)

        # Check files list
        if "files" in category_data:
            for file_info in category_data["files"]:
                if isinstance(file_info, dict):
                    file_info_copy = file_info.copy()
                    file_info_copy["category"] = category
                    files_list.append(file_info_copy)
                elif isinstance(file_info, str):
                    files_list.append({"name": file_info, "category": category})

        # Add directory information if present
        if "directory" in category_data:
            for file_info in files_list:
                file_info["directory"] = category_data["directory"]

        return files_list

    def validate_files(self) -> Dict[str, List[str]]:
        """
        Validate that all indexed files exist.

        Returns:
            Dictionary with 'missing' and 'found' keys containing file lists
        """
        found_files = []
        missing_files = []

        data_index = self.index_data.get("data_index", {})

        for category, category_data in data_index.items():
            if not isinstance(category_data, dict):
                continue

            # Check single file
            if "file" in category_data:
                file_path = self._resolve_path(category_data["file"])
                if file_path.exists():
                    found_files.append(str(file_path))
                else:
                    missing_files.append(str(file_path))

            # Check files in directory
            if "directory" in category_data:
                directory = category_data["directory"]
                if "files" in category_data:
                    for file_info in category_data["files"]:
                        if isinstance(file_info, dict):
                            file_name = file_info.get("name")
                            if file_name:
                                file_path = self._resolve_path(directory, file_name)
                                if file_path.exists():
                                    found_files.append(str(file_path))
                                else:
                                    missing_files.append(str(file_path))
                        elif isinstance(file_info, str):
                            file_path = self._resolve_path(directory, file_info)
                            if file_path.exists():
                                found_files.append(str(file_path))
                            else:
                                missing_files.append(str(file_path))

            # Check files list without directory
            elif "files" in category_data:
                for file_info in category_data["files"]:
                    if isinstance(file_info, dict):
                        file_name = file_info.get("name")
                        if file_name:
                            file_path = self._resolve_path(file_name)
                            if file_path.exists():
                                found_files.append(str(file_path))
                            else:
                                missing_files.append(str(file_path))
                    elif isinstance(file_info, str):
                        file_path = self._resolve_path(file_info)
                        if file_path.exists():
                            found_files.append(str(file_path))
                        else:
                            missing_files.append(str(file_path))

        return {"found": found_files, "missing": missing_files}

    def get_usage_guidelines(self, category: str) -> Optional[Dict[str, Any]]:
        """
        Get usage guidelines for a data category.

        Args:
            category: Data category name (can be specific category or
                usage guideline key like 'cmb_verification')

        Returns:
            Usage guidelines dictionary, or None if not found
        """
        # First check usage_guidelines section
        usage_guidelines = self.index_data.get("usage_guidelines", {})
        if category in usage_guidelines:
            return usage_guidelines[category]

        # Check if category exists in data_index
        data_index = self.index_data.get("data_index", {})
        category_data = data_index.get(category)

        if category_data is None:
            return None

        # Build usage info from category data
        usage_info: Dict[str, Any] = {}

        if "description" in category_data:
            usage_info["description"] = category_data["description"]

        if "usage" in category_data:
            usage_info["usage"] = category_data["usage"]

        # Get primary data files
        primary_data = []
        if "file" in category_data:
            primary_data.append(category_data["file"])
        if "files" in category_data:
            for file_info in category_data["files"]:
                if isinstance(file_info, dict):
                    primary_data.append(file_info.get("name", ""))
                elif isinstance(file_info, str):
                    primary_data.append(file_info)

        if primary_data:
            usage_info["primary_data"] = primary_data

        return usage_info if usage_info else None

    def search_by_name(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Search for files by name pattern.

        Args:
            pattern: Search pattern (supports wildcards like * and ?)

        Returns:
            List of matching file information dictionaries
        """
        matches = []
        data_index = self.index_data.get("data_index", {})

        for category, category_data in data_index.items():
            if not isinstance(category_data, dict):
                continue

            # Check single file
            if "file" in category_data:
                file_name = category_data["file"]
                if fnmatch.fnmatch(file_name, pattern):
                    file_info = {
                        "name": file_name,
                        "category": category,
                    }
                    if "size" in category_data:
                        file_info["size"] = category_data["size"]
                    if "description" in category_data:
                        file_info["description"] = category_data["description"]
                    matches.append(file_info)

            # Check files list
            if "files" in category_data:
                for file_info in category_data["files"]:
                    if isinstance(file_info, dict):
                        file_name = file_info.get("name", "")
                        if fnmatch.fnmatch(file_name, pattern):
                            file_info_copy = file_info.copy()
                            file_info_copy["category"] = category
                            matches.append(file_info_copy)
                    elif isinstance(file_info, str):
                        if fnmatch.fnmatch(file_info, pattern):
                            matches.append({"name": file_info, "category": category})

        return matches

    def _resolve_path(self, *path_parts: str) -> Path:
        """
        Resolve file path relative to data index root.

        Args:
            *path_parts: Path components to join

        Returns:
            Resolved Path object
        """
        if not path_parts:
            return self._data_index_root

        # Join path parts
        path_str = str(Path(*path_parts))

        # If path is absolute, return as is
        if Path(path_str).is_absolute():
            return Path(path_str)

        # Otherwise, resolve relative to data index root
        return (self._data_index_root / path_str).resolve()
