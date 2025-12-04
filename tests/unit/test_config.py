"""
Unit tests for configuration management module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from config import (
    Config,
    PhysicalConstants,
    DataPaths,
    LoggingConfig,
    get_config,
    initialize_config,
)


class TestPhysicalConstants:
    """Test physical constants validation."""

    def test_all_constants_defined(self):
        """Verify all physical constants are defined."""
        constants = PhysicalConstants(
            D=1.0e-8,
            z_CMB=1100.0,
            T_0=2.725,
            omega_CMB=1.0e11,
            arcmin_to_pc_at_z1100=60.0,
            delta_T_min=20.0,
            delta_T_max=30.0,
        )

        assert constants.D > 0
        assert constants.z_CMB == 1100.0
        assert constants.T_0 == 2.725
        assert constants.omega_CMB == 1.0e11
        assert constants.arcmin_to_pc_at_z1100 > 0
        assert constants.delta_T_min >= 0
        assert constants.delta_T_max >= constants.delta_T_min

    def test_z_cmb_range(self):
        """Validate z_CMB is within expected range."""
        constants = PhysicalConstants(
            D=1.0e-8,
            z_CMB=1100.0,
            T_0=2.725,
            omega_CMB=1.0e11,
            arcmin_to_pc_at_z1100=60.0,
            delta_T_min=20.0,
            delta_T_max=30.0,
        )

        assert 1000 <= constants.z_CMB <= 1200

    def test_temperature_range(self):
        """Verify temperature fluctuation range."""
        constants = PhysicalConstants(
            D=1.0e-8,
            z_CMB=1100.0,
            T_0=2.725,
            omega_CMB=1.0e11,
            arcmin_to_pc_at_z1100=60.0,
            delta_T_min=20.0,
            delta_T_max=30.0,
        )

        assert 20.0 <= constants.delta_T_min <= 30.0
        assert constants.delta_T_max >= constants.delta_T_min

    def test_constants_immutable(self):
        """Test that constants are immutable after initialization."""
        constants = PhysicalConstants(
            D=1.0e-8,
            z_CMB=1100.0,
            T_0=2.725,
            omega_CMB=1.0e11,
            arcmin_to_pc_at_z1100=60.0,
            delta_T_min=20.0,
            delta_T_max=30.0,
        )

        # Dataclass with frozen=True should be immutable
        with pytest.raises(Exception):
            constants.D = 2.0e-8


class TestDataPaths:
    """Test data paths configuration."""

    def test_all_paths_defined(self):
        """Verify all data paths are correctly set."""
        project_root = Path.cwd()
        paths = DataPaths(
            project_root=project_root,
            data_in=project_root / "data" / "in",
            data_theta=project_root / "data" / "theta",
            data_out=project_root / "data" / "out",
            data_tmp=project_root / "data" / "tmp",
            data_index=project_root / "data" / "in" / "data_index.yaml",
        )

        assert paths.project_root == project_root
        assert "data/in" in str(paths.data_in)
        assert "data/theta" in str(paths.data_theta)
        assert "data/out" in str(paths.data_out)
        assert "data/tmp" in str(paths.data_tmp)
        assert "data_index.yaml" in str(paths.data_index)

    def test_paths_can_be_created(self):
        """Check that paths can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            paths = DataPaths(
                project_root=project_root,
                data_in=project_root / "data" / "in",
                data_theta=project_root / "data" / "theta",
                data_out=project_root / "data" / "out",
                data_tmp=project_root / "data" / "tmp",
                data_index=project_root / "data" / "in" / "data_index.yaml",
            )

            # Create directories
            paths.data_out.mkdir(parents=True, exist_ok=True)
            paths.data_tmp.mkdir(parents=True, exist_ok=True)

            assert paths.data_out.exists()
            assert paths.data_tmp.exists()

    def test_path_resolution(self):
        """Test path resolution relative to project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            paths = DataPaths(
                project_root=project_root,
                data_in=project_root / "data" / "in",
                data_theta=project_root / "data" / "theta",
                data_out=project_root / "data" / "out",
                data_tmp=project_root / "data" / "tmp",
                data_index=project_root / "data" / "in" / "data_index.yaml",
            )

            # All paths should be absolute or relative to project_root
            assert paths.data_in.is_absolute() or str(paths.data_in).startswith(
                str(project_root)
            )


class TestConfigurationLoading:
    """Test configuration loading from YAML."""

    def test_load_defaults(self):
        """Test default configuration when no file provided."""
        config = Config.load()
        assert config is not None
        assert config.constants.z_CMB == 1100.0
        assert config.constants.T_0 == 2.725

    def test_load_from_valid_yaml(self):
        """Test loading from valid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "constants": {
                    "D": 1.0e-8,
                    "z_CMB": 1100.0,
                    "T_0": 2.725,
                    "omega_CMB": 1.0e11,
                    "arcmin_to_pc_at_z1100": 60.0,
                    "delta_T_min": 20.0,
                    "delta_T_max": 30.0,
                },
                "paths": {
                    "project_root": str(Path.cwd()),
                    "data_in": "data/in",
                    "data_theta": "data/theta",
                    "data_out": "data/out",
                    "data_tmp": "data/tmp",
                    "data_index": "data/in/data_index.yaml",
                },
                "logging": {
                    "level": "INFO",
                    "log_file": "logs/test.log",
                    "max_bytes": 10485760,
                    "backup_count": 5,
                },
            }
            yaml.dump(config_data, f)
            config_file = Path(f.name)

        try:
            config = Config.load(config_file)
            assert config.constants.z_CMB == 1100.0
            assert config.constants.T_0 == 2.725
        finally:
            config_file.unlink()

    def test_load_invalid_yaml(self):
        """Test error handling for invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_file = Path(f.name)

        try:
            with pytest.raises(ValueError):
                Config.load(config_file)
        finally:
            config_file.unlink()

    def test_load_missing_file(self):
        """Test error handling for missing config file."""
        config_file = Path("/nonexistent/config.yaml")
        with pytest.raises(FileNotFoundError):
            Config.load(config_file)

    def test_load_with_missing_fields(self):
        """Test loading with missing required fields (uses defaults)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {"constants": {}, "paths": {}, "logging": {}}
            yaml.dump(config_data, f)
            config_file = Path(f.name)

        try:
            config = Config.load(config_file)
            # Should use defaults
            assert config.constants.z_CMB == 1100.0
            assert config.constants.T_0 == 2.725
        finally:
            config_file.unlink()

    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = Config.load()
        config.validate()  # Should not raise

        # Invalid config (negative D)
        with pytest.raises(ValueError):
            invalid_constants = PhysicalConstants(
                D=-1.0,  # Invalid
                z_CMB=1100.0,
                T_0=2.725,
                omega_CMB=1.0e11,
                arcmin_to_pc_at_z1100=60.0,
                delta_T_min=20.0,
                delta_T_max=30.0,
            )
            invalid_paths = DataPaths(
                project_root=Path.cwd(),
                data_in=Path.cwd() / "data" / "in",
                data_theta=Path.cwd() / "data" / "theta",
                data_out=Path.cwd() / "data" / "out",
                data_tmp=Path.cwd() / "data" / "tmp",
                data_index=Path.cwd() / "data" / "in" / "data_index.yaml",
            )
            invalid_logging = LoggingConfig(
                level="INFO",
                log_file=None,
                max_bytes=10485760,
                backup_count=5,
            )
            invalid_config = Config(invalid_constants, invalid_paths, invalid_logging)
            invalid_config.validate()


class TestConfigurationAccess:
    """Test configuration access methods."""

    def test_get_config_before_initialization(self):
        """Test get_config() raises error before initialization."""
        # Reset global instance
        Config._instance = None
        with pytest.raises(RuntimeError):
            get_config()

    def test_initialize_config(self):
        """Test initialize_config() sets global config."""
        Config._instance = None
        config = initialize_config()
        assert config is not None
        assert Config._instance is not None
        assert get_config() is config

    def test_config_methods_accessible(self):
        """Test Config class interface (all methods accessible)."""
        config = Config.load()
        assert hasattr(config, "validate")
        assert hasattr(config, "setup_logging")
        assert hasattr(config, "constants")
        assert hasattr(config, "paths")
        assert hasattr(config, "logging_config")

    def test_type_safety(self):
        """Verify type hints work correctly."""
        config = Config.load()
        assert isinstance(config.constants, PhysicalConstants)
        assert isinstance(config.paths, DataPaths)
        assert isinstance(config.logging_config, LoggingConfig)

    def test_cmb_config_loading(self):
        """Test that CMB config is loaded if available."""
        config = Config.load()
        # CMB config should be loaded if cmb_config.yaml exists
        assert hasattr(config, "cmb_config")
        assert isinstance(config.cmb_config, dict)


class TestLoggingConfig:
    """Test logging configuration."""

    def test_logging_setup(self):
        """Test logging setup."""
        config = Config.load()
        # Should not raise
        config.setup_logging()

    def test_logging_levels(self):
        """Test different logging levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            logging_config = LoggingConfig(
                level=level,
                log_file=None,
                max_bytes=10485760,
                backup_count=5,
            )
            assert logging_config.level == level

    def test_invalid_logging_level(self):
        """Test validation of invalid logging level."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "constants": {
                    "D": 1.0e-8,
                    "z_CMB": 1100.0,
                    "T_0": 2.725,
                    "omega_CMB": 1.0e11,
                    "arcmin_to_pc_at_z1100": 60.0,
                    "delta_T_min": 20.0,
                    "delta_T_max": 30.0,
                },
                "paths": {
                    "project_root": str(Path.cwd()),
                },
                "logging": {
                    "level": "INVALID_LEVEL",
                    "max_bytes": 10485760,
                    "backup_count": 5,
                },
            }
            yaml.dump(config_data, f)
            config_file = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid logging level"):
                Config.load(config_file)
        finally:
            config_file.unlink()
