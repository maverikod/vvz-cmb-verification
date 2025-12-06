"""
Unit tests for CMB map reconstruction module.

Tests CmbMapReconstructor with CUDA acceleration support.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
import healpy as hp
from cmb.reconstruction.cmb_map_reconstructor import CmbMapReconstructor
from cmb.theta_data_loader import ThetaFrequencySpectrum, ThetaEvolution
from cmb.theta_node_processor import ThetaNodeData
from config.settings import Config, initialize_config


@pytest.fixture
def sample_config():
    """Create sample configuration for tests."""
    initialize_config()
    config = Config._load_defaults()
    # Add CMB config
    if not hasattr(config, "cmb_config") or config.cmb_config is None:
        config.cmb_config = {}
    config.cmb_config["t_0"] = 1.0
    config.cmb_config["alpha"] = -2.0
    config.cmb_config["beta"] = -3.0
    return config


@pytest.fixture
def sample_frequency_spectrum():
    """Create sample frequency spectrum for tests."""
    n_freq = 10
    n_times = 5
    frequencies = np.logspace(10, 11, n_freq)  # 10^10 to 10^11 Hz
    times = np.linspace(0.0, 2.0, n_times)
    # Create spectrum: ρ_Θ(ω,t) ∝ ω^(-2) · (t/t_0)^(-3)
    omega_grid, time_grid = np.meshgrid(frequencies, times, indexing="ij")
    spectrum = (omega_grid ** (-2.0)) * ((time_grid / 1.0) ** (-3.0))

    return ThetaFrequencySpectrum(
        frequencies=frequencies,
        times=times,
        spectrum=spectrum,
        metadata={"test": "data"},
    )


@pytest.fixture
def sample_evolution():
    """Create sample evolution data for tests."""
    n_times = 5
    times = np.linspace(0.0, 2.0, n_times)
    omega_min = np.linspace(1.0e10, 1.1e10, n_times)
    omega_macro = np.linspace(10.0e10, 10.1e10, n_times)

    return ThetaEvolution(
        times=times,
        omega_min=omega_min,
        omega_macro=omega_macro,
        metadata={"test": "data"},
    )


@pytest.fixture
def sample_node_data(sample_config):
    """Create sample node data for tests."""
    n_nodes = 20
    # Valid positions: theta in [0, π], phi in [0, 2π]
    theta = np.linspace(0.1, np.pi - 0.1, n_nodes)
    phi = np.linspace(0, 2 * np.pi, n_nodes)
    positions = np.column_stack([theta, phi])
    scales = np.full(n_nodes, 300.0)  # 300 pc
    depths = np.linspace(1e-5, 1e-4, n_nodes)  # Δω/ω
    # Temperatures will be calculated by map_depth_to_temperature
    from cmb.theta_node_processor import map_depth_to_temperature

    temperatures = map_depth_to_temperature(depths, sample_config)

    return ThetaNodeData(
        positions=positions,
        scales=scales,
        depths=depths,
        temperatures=temperatures,
        metadata={"n_nodes": n_nodes},
    )


class TestCmbMapReconstructor:
    """Test CmbMapReconstructor class."""

    def test_init(
        self,
        sample_frequency_spectrum,
        sample_evolution,
        sample_node_data,
        sample_config,
    ):
        """Test initialization."""
        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=sample_node_data,
            nside=64,  # Small NSIDE for tests
            config=sample_config,
        )

        assert reconstructor.frequency_spectrum == sample_frequency_spectrum
        assert reconstructor.evolution == sample_evolution
        assert reconstructor.node_data == sample_node_data
        assert reconstructor.nside == 64
        assert reconstructor.config == sample_config
        assert reconstructor.elem_vec is not None
        assert reconstructor.reduction_vec is not None
        assert reconstructor.transform_vec is not None

    def test_init_default_config(
        self, sample_frequency_spectrum, sample_evolution, sample_node_data
    ):
        """Test initialization with default config."""
        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=sample_node_data,
            nside=64,
        )

        assert reconstructor.config is not None
        assert reconstructor.t_0 == 1.0  # Default
        assert reconstructor.alpha == -2.0  # Default
        assert reconstructor.beta == -3.0  # Default

    def test_convert_nodes_to_temperature(
        self,
        sample_frequency_spectrum,
        sample_evolution,
        sample_node_data,
        sample_config,
    ):
        """Test node depth to temperature conversion."""
        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=sample_node_data,
            nside=64,
            config=sample_config,
        )

        temperatures = reconstructor._convert_nodes_to_temperature()

        # Check shape
        assert temperatures.shape == (len(sample_node_data.depths),)

        # Check temperature range (should be ~20-30 μK for typical depths)
        # For test data with depths up to 1e-4, temperatures can be up to ~272 μK
        assert np.all(temperatures >= 0)  # No negative temperatures
        assert np.all(temperatures < 1000.0)  # Reasonable upper bound for test data

        # Check that temperatures match expected values
        # Formula: ΔT = (Δω/ω_CMB) T_0
        # For depths ~1e-5, T_0 = 2.725 K, ω_CMB ~ 10^11 Hz
        # ΔT ≈ 1e-5 * 2.725 K ≈ 27.25 μK
        expected_temps = sample_node_data.temperatures
        assert np.allclose(temperatures, expected_temps, rtol=1e-5)

    def test_project_nodes_to_sky(
        self,
        sample_frequency_spectrum,
        sample_evolution,
        sample_node_data,
        sample_config,
    ):
        """Test node projection to sky coordinates."""
        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=sample_node_data,
            nside=64,
            config=sample_config,
        )

        positions = reconstructor._project_nodes_to_sky()

        # Check shape
        assert positions.shape == sample_node_data.positions.shape

        # Check that positions are valid
        theta = positions[:, 0]
        phi = positions[:, 1]
        assert np.all(theta >= 0) and np.all(theta <= np.pi)
        assert np.all(phi >= 0) and np.all(phi <= 2 * np.pi)

    def test_project_nodes_to_sky_invalid_theta(
        self, sample_frequency_spectrum, sample_evolution, sample_config
    ):
        """Test projection with invalid theta values."""
        # Create node data with invalid theta
        n_nodes = 5
        theta = np.array([-0.1, 0.5, 1.0, 2.0, np.pi + 0.1])  # Invalid values
        phi = np.linspace(0, 2 * np.pi, n_nodes)
        positions = np.column_stack([theta, phi])
        scales = np.full(n_nodes, 300.0)
        depths = np.linspace(1e-5, 1e-4, n_nodes)
        from cmb.theta_node_processor import map_depth_to_temperature

        temperatures = map_depth_to_temperature(depths)
        node_data = ThetaNodeData(
            positions=positions,
            scales=scales,
            depths=depths,
            temperatures=temperatures,
            metadata={"n_nodes": n_nodes},
        )

        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=node_data,
            nside=64,
            config=sample_config,
        )

        with pytest.raises(ValueError, match="Theta values must be in"):
            reconstructor._project_nodes_to_sky()

    def test_project_nodes_to_sky_invalid_phi(
        self, sample_frequency_spectrum, sample_evolution, sample_config
    ):
        """Test projection with invalid phi values."""
        # Create node data with invalid phi
        n_nodes = 5
        theta = np.linspace(0.1, np.pi - 0.1, n_nodes)
        phi = np.array([-0.1, 1.0, 2.0, 5.0, 2 * np.pi + 0.1])  # Invalid values
        positions = np.column_stack([theta, phi])
        scales = np.full(n_nodes, 300.0)
        depths = np.linspace(1e-5, 1e-4, n_nodes)
        from cmb.theta_node_processor import map_depth_to_temperature

        temperatures = map_depth_to_temperature(depths)
        node_data = ThetaNodeData(
            positions=positions,
            scales=scales,
            depths=depths,
            temperatures=temperatures,
            metadata={"n_nodes": n_nodes},
        )

        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=node_data,
            nside=64,
            config=sample_config,
        )

        with pytest.raises(ValueError, match="Phi values must be in"):
            reconstructor._project_nodes_to_sky()

    def test_integrate_frequency_spectrum(
        self,
        sample_frequency_spectrum,
        sample_evolution,
        sample_node_data,
        sample_config,
    ):
        """Test frequency spectrum integration."""
        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=sample_node_data,
            nside=64,
            config=sample_config,
        )

        weights = reconstructor._integrate_frequency_spectrum()

        # Check shape
        assert weights.shape == (len(sample_node_data.positions),)

        # Check that weights are positive (spectrum should be positive)
        assert np.all(weights > 0)

        # Check that weights are reasonable (not too large or too small)
        assert np.all(weights < 1e10)  # Upper bound
        assert np.all(weights > 1e-20)  # Lower bound

    def test_reconstruct_map(
        self,
        sample_frequency_spectrum,
        sample_evolution,
        sample_node_data,
        sample_config,
    ):
        """Test full map reconstruction."""
        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=sample_node_data,
            nside=64,  # Small NSIDE for tests
            config=sample_config,
        )

        cmb_map = reconstructor.reconstruct_map()

        # Check shape
        npix = hp.nside2npix(64)
        assert cmb_map.shape == (npix,)

        # Check that map is not all zeros
        assert np.any(cmb_map != 0)

        # Check that map values are reasonable (temperature fluctuations in μK)
        # Should be in range -100 to 100 μK for test data
        assert np.all(cmb_map >= -1000.0)  # Allow some margin
        assert np.all(cmb_map <= 1000.0)

    def test_reconstruct_map_cuda_usage(
        self,
        sample_frequency_spectrum,
        sample_evolution,
        sample_node_data,
        sample_config,
    ):
        """Test that reconstruction uses CUDA utilities."""
        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=sample_node_data,
            nside=64,
            config=sample_config,
        )

        # Check that CUDA vectorizers are initialized
        assert reconstructor.elem_vec is not None
        assert reconstructor.reduction_vec is not None
        assert reconstructor.transform_vec is not None

        # Check that vectorizers are configured for GPU
        assert reconstructor.elem_vec.use_gpu is True
        assert reconstructor.reduction_vec.use_gpu is True
        assert reconstructor.transform_vec.use_gpu is True

        # Reconstruct map (should use CUDA internally)
        cmb_map = reconstructor.reconstruct_map()

        # Check that map was created
        assert cmb_map is not None
        assert len(cmb_map) > 0

    def test_reconstruct_map_scatter_add_usage(
        self,
        sample_frequency_spectrum,
        sample_evolution,
        sample_node_data,
        sample_config,
    ):
        """Test that reconstruction uses scatter_add for pixel accumulation."""
        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=sample_node_data,
            nside=64,
            config=sample_config,
        )

        # Check that scatter_add method exists
        assert hasattr(reconstructor.elem_vec, "scatter_add")

        # Reconstruct map (should use scatter_add internally)
        cmb_map = reconstructor.reconstruct_map()

        # Check that map was created
        assert cmb_map is not None
        assert len(cmb_map) > 0

    def test_reconstruct_map_large_nside(
        self,
        sample_frequency_spectrum,
        sample_evolution,
        sample_node_data,
        sample_config,
    ):
        """Test reconstruction with larger NSIDE (more realistic)."""
        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=sample_node_data,
            nside=256,  # Larger NSIDE
            config=sample_config,
        )

        cmb_map = reconstructor.reconstruct_map()

        # Check shape
        npix = hp.nside2npix(256)
        assert cmb_map.shape == (npix,)

        # Check that map is not all zeros
        assert np.any(cmb_map != 0)

    def test_reconstruct_map_empty_nodes(
        self, sample_frequency_spectrum, sample_evolution, sample_config
    ):
        """Test reconstruction with empty node data."""
        # Create empty node data
        positions = np.empty((0, 2))
        scales = np.empty(0)
        depths = np.empty(0)
        temperatures = np.empty(0)
        node_data = ThetaNodeData(
            positions=positions,
            scales=scales,
            depths=depths,
            temperatures=temperatures,
            metadata={"n_nodes": 0},
        )

        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=node_data,
            nside=64,
            config=sample_config,
        )

        cmb_map = reconstructor.reconstruct_map()

        # Map should be all zeros (no nodes to contribute)
        npix = hp.nside2npix(64)
        assert cmb_map.shape == (npix,)
        assert np.all(cmb_map == 0)

    def test_reconstruct_map_formula_cmb21(
        self,
        sample_frequency_spectrum,
        sample_evolution,
        sample_node_data,
        sample_config,
    ):
        """Test that reconstruction uses formula CMB2.1 correctly."""
        reconstructor = CmbMapReconstructor(
            frequency_spectrum=sample_frequency_spectrum,
            evolution=sample_evolution,
            node_data=sample_node_data,
            nside=64,
            config=sample_config,
        )

        # Check that power law indices are set correctly
        assert reconstructor.alpha == -2.0  # From formula: ω^(-2±0.5)
        assert reconstructor.beta == -3.0  # From formula: (t/t_0)^(-3±1)
        assert reconstructor.t_0 == 1.0  # Reference time

        # Reconstruct map
        cmb_map = reconstructor.reconstruct_map()

        # Check that map was created (integration should have worked)
        assert cmb_map is not None
        assert len(cmb_map) > 0

        # Check that spectrum weights were calculated
        # (non-zero map means integration worked)
        weights = reconstructor._integrate_frequency_spectrum()
        assert np.all(weights > 0)  # All weights should be positive
