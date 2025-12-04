"""
Main configuration module for CMB verification project.

This module provides centralized configuration management including
physical constants, data paths, and project-wide settings.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from logging.handlers import RotatingFileHandler
import yaml


@dataclass(frozen=True)
class PhysicalConstants:
    """
    Physical constants used in CMB verification calculations.

    Attributes:
        D: Distance parameter for frequency-to-multipole conversion
        z_CMB: Redshift of CMB surface (approximately 1100)
        T_0: CMB temperature in Kelvin (2.725 K)
        omega_CMB: CMB frequency in Hz (~10^11 Hz)
        arcmin_to_pc_at_z1100: Conversion factor from arcmin to parsec
            at z≈1100
        delta_T_min: Minimum temperature fluctuation in μK
        delta_T_max: Maximum temperature fluctuation in μK
    """

    D: float
    z_CMB: float
    T_0: float
    omega_CMB: float
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

    # Global configuration instance
    _instance: Optional["Config"] = None

    def __init__(
        self,
        constants: PhysicalConstants,
        paths: DataPaths,
        logging_config: LoggingConfig,
        cmb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize configuration.

        Args:
            constants: Physical constants configuration
            paths: Data paths configuration
            logging_config: Logging configuration
            cmb_config: CMB-specific configuration dictionary
        """
        self.constants = constants
        self.paths = paths
        self.logging_config = logging_config
        self.cmb_config = cmb_config or {}

    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> "Config":
        """
        Load configuration from YAML file or use defaults.

        Args:
            config_file: Path to configuration YAML file.
                If None, uses defaults.

        Returns:
            Config instance with loaded settings

        Raises:
            FileNotFoundError: If config_file is specified but doesn't exist
            ValueError: If configuration values are invalid
            yaml.YAMLError: If YAML file is malformed
        """
        if config_file is None:
            return cls._load_defaults()

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}") from e

        # Load CMB-specific configuration
        cmb_config_file = config_file.parent / "cmb_config.yaml"
        cmb_config = None
        if cmb_config_file.exists():
            try:
                with open(cmb_config_file, "r", encoding="utf-8") as f:
                    cmb_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                # Log warning but don't fail if CMB config is invalid
                logging.warning(f"Could not load CMB config: {e}")

        # Extract configuration sections
        constants_data = config_data.get("constants", {})
        paths_data = config_data.get("paths", {})
        logging_data = config_data.get("logging", {})

        # Create PhysicalConstants
        constants = PhysicalConstants(
            D=constants_data.get("D", 1.0e-8),  # Default calibration parameter
            z_CMB=constants_data.get("z_CMB", 1100.0),
            T_0=constants_data.get("T_0", 2.725),
            omega_CMB=constants_data.get("omega_CMB", 1.0e11),
            arcmin_to_pc_at_z1100=constants_data.get("arcmin_to_pc_at_z1100", 60.0),
            delta_T_min=constants_data.get("delta_T_min", 20.0),
            delta_T_max=constants_data.get("delta_T_max", 30.0),
        )

        # Create DataPaths
        project_root = Path(paths_data.get("project_root", Path.cwd()))
        paths = DataPaths(
            project_root=project_root,
            data_in=project_root / paths_data.get("data_in", "data/in"),
            data_theta=project_root / paths_data.get("data_theta", "data/theta"),
            data_out=project_root / paths_data.get("data_out", "data/out"),
            data_tmp=project_root / paths_data.get("data_tmp", "data/tmp"),
            data_index=project_root
            / paths_data.get("data_index", "data/in/data_index.yaml"),
        )

        # Create LoggingConfig
        log_file = None
        if logging_data.get("log_file"):
            log_file = project_root / logging_data["log_file"]

        logging_config = LoggingConfig(
            level=logging_data.get("level", "INFO"),
            log_file=log_file,
            max_bytes=logging_data.get("max_bytes", 10 * 1024 * 1024),  # 10 MB
            backup_count=logging_data.get("backup_count", 5),
        )

        config = cls(constants, paths, logging_config, cmb_config)
        config.validate()
        return config

    @classmethod
    def _load_defaults(cls) -> "Config":
        """
        Create configuration with default values.

        Returns:
            Config instance with default settings
        """
        # Determine project root (parent of config directory)
        project_root = Path(__file__).parent.parent

        # Default physical constants
        # D: calibration parameter for l ≈ π D ω
        # For l_max=10000 and ω_max=350 GHz,
        # D should give reasonable conversion
        # Using D ≈ 1.0e-8 as default (will be calibrated)
        constants = PhysicalConstants(
            D=1.0e-8,
            z_CMB=1100.0,
            T_0=2.725,  # K
            omega_CMB=1.0e11,  # Hz
            arcmin_to_pc_at_z1100=60.0,  # 2-5 arcmin → 100-300 pc
            delta_T_min=20.0,  # μK
            delta_T_max=30.0,  # μK
        )

        # Default paths
        paths = DataPaths(
            project_root=project_root,
            data_in=project_root / "data" / "in",
            data_theta=project_root / "data" / "theta",
            data_out=project_root / "data" / "out",
            data_tmp=project_root / "data" / "tmp",
            data_index=project_root / "data" / "in" / "data_index.yaml",
        )

        # Default logging
        logging_config = LoggingConfig(
            level="INFO",
            log_file=project_root / "logs" / "cmb_verification.log",
            max_bytes=10 * 1024 * 1024,  # 10 MB
            backup_count=5,
        )

        # Try to load CMB config if it exists
        cmb_config_file = project_root / "config" / "cmb_config.yaml"
        cmb_config = None
        if cmb_config_file.exists():
            try:
                with open(cmb_config_file, "r", encoding="utf-8") as f:
                    cmb_config = yaml.safe_load(f)
            except (yaml.YAMLError, FileNotFoundError):
                # Ignore errors, use None
                pass

        config = cls(constants, paths, logging_config, cmb_config)
        config.validate()
        return config

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid
        """
        # Validate physical constants
        if self.constants.D <= 0:
            raise ValueError("D must be positive")
        if not (1000 <= self.constants.z_CMB <= 1200):
            raise ValueError(f"z_CMB should be around 1100, got {self.constants.z_CMB}")
        if self.constants.T_0 <= 0:
            raise ValueError("T_0 must be positive")
        if self.constants.omega_CMB <= 0:
            raise ValueError("omega_CMB must be positive")
        if self.constants.arcmin_to_pc_at_z1100 <= 0:
            raise ValueError("arcmin_to_pc_at_z1100 must be positive")
        if self.constants.delta_T_min < 0:
            raise ValueError("delta_T_min must be non-negative")
        if self.constants.delta_T_max < self.constants.delta_T_min:
            raise ValueError("delta_T_max must be >= delta_T_min")

        # Validate paths
        if not self.paths.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.paths.project_root}")

        # Validate logging
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging_config.level.upper() not in valid_levels:
            raise ValueError(
                f"Invalid logging level: {self.logging_config.level}. "
                f"Must be one of {valid_levels}"
            )
        if self.logging_config.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.logging_config.backup_count < 0:
            raise ValueError("backup_count must be non-negative")

    def setup_logging(self) -> None:
        """
        Set up logging based on configuration.

        Configures Python logging module with settings from
        logging_config.
        """
        # Get logging level
        level = getattr(logging, self.logging_config.level.upper(), logging.INFO)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler with rotation
        if self.logging_config.log_file:
            # Create log directory if it doesn't exist
            self.logging_config.log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                self.logging_config.log_file,
                maxBytes=self.logging_config.max_bytes,
                backupCount=self.logging_config.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)


def get_config() -> Config:
    """
    Get global configuration instance.

    Returns:
        Global Config instance

    Raises:
        RuntimeError: If configuration has not been initialized
    """
    if Config._instance is None:
        raise RuntimeError(
            "Configuration has not been initialized. " "Call initialize_config() first."
        )
    return Config._instance


def initialize_config(config_file: Optional[Path] = None) -> Config:
    """
    Initialize global configuration.

    Args:
        config_file: Path to configuration YAML file

    Returns:
        Initialized Config instance
    """
    Config._instance = Config.load(config_file)
    Config._instance.setup_logging()
    return Config._instance
