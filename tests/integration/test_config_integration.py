"""
Integration tests for configuration management.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from config import Config, initialize_config, get_config


class TestConfigurationIntegration:
    """Integration tests for configuration usage."""

    def test_import_and_use_config(self):
        """Test that other modules can import and use config."""
        from config import Config, PhysicalConstants, DataPaths

        config = Config.load()
        assert config is not None
        assert isinstance(config.constants, PhysicalConstants)
        assert isinstance(config.paths, DataPaths)

    def test_global_config_access(self):
        """Verify configuration is accessible throughout project."""
        Config._instance = None
        config1 = initialize_config()
        config2 = get_config()

        assert config1 is config2
        assert Config._instance is config1

    def test_configuration_consistency(self):
        """Verify configuration consistency across modules."""
        Config._instance = None
        initialize_config()

        # Access from different "modules" (different calls)
        config1 = get_config()
        config2 = get_config()

        assert config1.constants.z_CMB == config2.constants.z_CMB
        assert config1.paths.data_in == config2.paths.data_in
        assert config1.logging_config.level == config2.logging_config.level

    def test_config_with_actual_paths(self):
        """Test configuration with actual data paths and constants."""
        config = Config.load()

        # Verify paths are set correctly
        assert config.paths.project_root.exists()
        assert "data" in str(config.paths.data_in)
        assert "data" in str(config.paths.data_out)

        # Verify constants are reasonable
        assert 1000 <= config.constants.z_CMB <= 1200
        assert config.constants.T_0 > 0
        assert config.constants.omega_CMB > 0
        assert config.constants.delta_T_min <= config.constants.delta_T_max

    def test_cmb_config_available(self):
        """Test that CMB-specific configuration is available."""
        config = Config.load()

        # CMB config should be a dictionary
        assert isinstance(config.cmb_config, dict)

        # If CMB config exists, it should have expected keys
        if config.cmb_config:
            expected_keys = [
                "frequency",
                "multipole",
                "node",
                "temperature",
                "power_spectrum",
            ]
            # At least some of these should be present
            assert any(key in config.cmb_config for key in expected_keys)
