"""
Main configuration module for CMB verification project.

This module provides centralized configuration management including
physical constants, data paths, and project-wide settings.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import yaml


@dataclass(frozen=True)
class PhysicalConstants:
    """
    Physical constants used in CMB verification calculations.
    
    Attributes:
        D: Distance parameter for frequency-to-multipole conversion
        z_CMB: Redshift of CMB surface (approximately 1100)
        arcmin_to_pc_at_z1100: Conversion factor from arcmin to parsec at z≈1100
        delta_T_min: Minimum temperature fluctuation in μK
        delta_T_max: Maximum temperature fluctuation in μK
    """
    D: float
    z_CMB: float
    arcmin_to_pc_at_z1100: float
    delta_T_min: float  # μK
    delta_T_max: float  # μK


@dataclass(frozen=True)
class DataPaths:
    """
    Data directory paths configuration.
    
    Attributes:
        project_root: Root directory of the project
        data_in: Input data directory
        data_theta: Θ-field data directory
        data_out: Output data directory
        data_tmp: Temporary data directory
        data_index: Path to data index YAML file
    """
    project_root: Path
    data_in: Path
    data_theta: Path
    data_out: Path
    data_tmp: Path
    data_index: Path


@dataclass(frozen=True)
class LoggingConfig:
    """
    Logging configuration.
    
    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        max_bytes: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
    """
    level: str
    log_file: Optional[Path]
    max_bytes: int
    backup_count: int


class Config:
    """
    Main configuration class for CMB verification project.
    
    Provides centralized access to all configuration settings including
    physical constants, data paths, and logging configuration.
    
    Usage:
        config = Config.load()
        print(config.constants.z_CMB)
        print(config.paths.data_in)
    """
    
    def __init__(
        self,
        constants: PhysicalConstants,
        paths: DataPaths,
        logging_config: LoggingConfig
    ):
        """
        Initialize configuration.
        
        Args:
            constants: Physical constants configuration
            paths: Data paths configuration
            logging_config: Logging configuration
        """
        self.constants = constants
        self.paths = paths
        self.logging_config = logging_config
    
    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> "Config":
        """
        Load configuration from YAML file or use defaults.
        
        Args:
            config_file: Path to configuration YAML file. If None, uses defaults.
            
        Returns:
            Config instance with loaded settings
            
        Raises:
            FileNotFoundError: If config_file is specified but doesn't exist
            ValueError: If configuration values are invalid
        """
        pass
    
    @classmethod
    def _load_defaults(cls) -> "Config":
        """
        Create configuration with default values.
        
        Returns:
            Config instance with default settings
        """
        pass
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        pass
    
    def setup_logging(self) -> None:
        """
        Set up logging based on configuration.
        
        Configures Python logging module with settings from
        logging_config.
        """
        pass


def get_config() -> Config:
    """
    Get global configuration instance.
    
    Returns:
        Global Config instance
        
    Raises:
        RuntimeError: If configuration has not been initialized
    """
    pass


def initialize_config(config_file: Optional[Path] = None) -> Config:
    """
    Initialize global configuration.
    
    Args:
        config_file: Path to configuration YAML file
        
    Returns:
        Initialized Config instance
    """
    pass

