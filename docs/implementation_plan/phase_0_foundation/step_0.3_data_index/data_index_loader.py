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
        pass

    def get_file_path(self, category: str, file_name: str) -> Optional[Path]:
        """
        Get file path for specific data file.

        Args:
            category: Data category (e.g., 'act_dr6_02', 'cmb_microcells')
            file_name: Name of the file

        Returns:
            Path to file, or None if not found
        """
        pass

    def get_files_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all files in a specific category.

        Args:
            category: Data category name

        Returns:
            List of file information dictionaries
        """
        pass

    def validate_files(self) -> Dict[str, List[str]]:
        """
        Validate that all indexed files exist.

        Returns:
            Dictionary with 'missing' and 'found' keys containing file lists
        """
        pass

    def get_usage_guidelines(self, category: str) -> Optional[Dict[str, Any]]:
        """
        Get usage guidelines for a data category.

        Args:
            category: Data category name

        Returns:
            Usage guidelines dictionary, or None if not found
        """
        pass

    def search_by_name(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Search for files by name pattern.

        Args:
            pattern: Search pattern (supports wildcards)

        Returns:
            List of matching file information dictionaries
        """
        pass
