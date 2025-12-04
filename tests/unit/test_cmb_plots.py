"""
Unit tests for cmb_plots module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import healpy as hp
import matplotlib

# Use non-interactive backend for tests
matplotlib.use("Agg")

from utils.visualization.cmb_plots import (
    plot_healpix_map,
    plot_temperature_fluctuations,
    plot_map_comparison,
    plot_node_overlay,
)


class TestPlotHealpixMap:
    """Tests for plot_healpix_map function."""

    def test_plot_valid_map(self):
        """Test plotting a valid HEALPix map."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "map.png"
            plot_healpix_map(test_map, output_path=output_path)

            assert output_path.exists()

    def test_plot_empty_map_raises_error(self):
        """Test plotting empty map raises ValueError."""
        empty_map = np.array([])

        with pytest.raises(ValueError, match="empty"):
            plot_healpix_map(empty_map)

    def test_plot_invalid_data_raises_error(self):
        """Test plotting invalid data raises ValueError."""
        with pytest.raises(ValueError, match="must be a numpy array"):
            plot_healpix_map([1, 2, 3])


class TestPlotTemperatureFluctuations:
    """Tests for plot_temperature_fluctuations function."""

    def test_plot_fluctuations(self):
        """Test plotting temperature fluctuations."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fluctuations.png"
            plot_temperature_fluctuations(test_map, output_path=output_path)

            assert output_path.exists()

    def test_plot_invalid_data_raises_error(self):
        """Test plotting invalid data raises ValueError."""
        with pytest.raises(ValueError, match="must be a numpy array"):
            plot_temperature_fluctuations([1, 2, 3])


class TestPlotMapComparison:
    """Tests for plot_map_comparison function."""

    def test_plot_comparison(self):
        """Test plotting map comparison."""
        nside = 64
        npix = hp.nside2npix(nside)
        map1 = np.random.randn(npix)
        map2 = np.random.randn(npix)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "comparison.png"
            plot_map_comparison(map1, map2, output_path=output_path)

            assert output_path.exists()

    def test_plot_comparison_different_sizes(self):
        """Test plotting comparison with different sizes raises ValueError."""
        map1 = np.random.randn(100)
        map2 = np.random.randn(200)

        with pytest.raises(ValueError, match="must match"):
            plot_map_comparison(map1, map2)

    def test_plot_comparison_invalid_data(self):
        """Test plotting comparison with invalid data raises ValueError."""
        map1 = np.random.randn(100)
        map2 = [1, 2, 3]

        with pytest.raises(ValueError, match="must be numpy arrays"):
            plot_map_comparison(map1, map2)


class TestPlotNodeOverlay:
    """Tests for plot_node_overlay function."""

    def test_plot_node_overlay(self):
        """Test plotting map with node overlay."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        # Create some node positions (theta, phi)
        n_nodes = 10
        node_positions = np.random.rand(n_nodes, 2)
        node_positions[:, 0] *= np.pi  # theta: 0 to π
        node_positions[:, 1] *= 2 * np.pi  # phi: 0 to 2π

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nodes.png"
            plot_node_overlay(test_map, node_positions, output_path=output_path)

            assert output_path.exists()

    def test_plot_node_overlay_invalid_shape(self):
        """Test plotting with invalid node positions shape raises ValueError."""
        nside = 64
        npix = hp.nside2npix(nside)
        test_map = np.random.randn(npix)

        # Invalid shape: should be (N, 2)
        node_positions = np.random.rand(10, 3)  # Wrong shape

        with pytest.raises(ValueError, match="shape"):
            plot_node_overlay(test_map, node_positions)

    def test_plot_node_overlay_invalid_map(self):
        """Test plotting with invalid map raises ValueError."""
        node_positions = np.random.rand(10, 2)

        with pytest.raises(ValueError, match="must be a numpy array"):
            plot_node_overlay([1, 2, 3], node_positions)
