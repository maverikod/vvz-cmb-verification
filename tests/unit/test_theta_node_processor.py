"""
Unit tests for Θ-node data processing module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
import csv
from cmb.theta_node_processor import (
    ThetaNodeData,
    map_depth_to_temperature,
    process_node_data,
    validate_node_data,
)
from cmb.theta_node_loader import load_node_geometry, load_node_depths
from config.settings import Config, initialize_config


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create sample configuration for tests."""
    config = Config._load_defaults()
    initialize_config()
    return config


@pytest.fixture
def sample_node_geometry_csv(temp_dir):
    """Create sample node geometry CSV file."""
    file_path = temp_dir / "node_geometry.csv"

    # Create sample data: theta (rad), phi (rad), scale (pc)
    n_nodes = 10
    theta = np.linspace(0.1, np.pi - 0.1, n_nodes)
    phi = np.linspace(0, 2 * np.pi, n_nodes)
    scales = np.full(n_nodes, 300.0)  # 300 pc

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["theta", "phi", "scale"])
        writer.writeheader()
        for i in range(n_nodes):
            writer.writerow(
                {
                    "theta": theta[i],
                    "phi": phi[i],
                    "scale": scales[i],
                }
            )

    return file_path


@pytest.fixture
def sample_node_geometry_json(temp_dir):
    """Create sample node geometry JSON file."""
    file_path = temp_dir / "node_geometry.json"

    n_nodes = 10
    theta = np.linspace(0.1, np.pi - 0.1, n_nodes)
    phi = np.linspace(0, 2 * np.pi, n_nodes)
    scales = np.full(n_nodes, 300.0)

    data = {
        "theta": theta.tolist(),
        "phi": phi.tolist(),
        "scale": scales.tolist(),
    }

    with open(file_path, "w") as f:
        json.dump(data, f)

    return file_path


@pytest.fixture
def sample_node_depths_csv(temp_dir):
    """Create sample node depths CSV file."""
    file_path = temp_dir / "node_depths.csv"

    n_nodes = 10
    # Typical depths: Δω/ω ~ 10^-5 to 10^-4
    depths = np.linspace(1e-5, 1e-4, n_nodes)

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["depth"])
        writer.writeheader()
        for depth in depths:
            writer.writerow({"depth": depth})

    return file_path


@pytest.fixture
def sample_node_depths_json(temp_dir):
    """Create sample node depths JSON file."""
    file_path = temp_dir / "node_depths.json"

    n_nodes = 10
    depths = np.linspace(1e-5, 1e-4, n_nodes)

    data = {"depths": depths.tolist()}

    with open(file_path, "w") as f:
        json.dump(data, f)

    return file_path


class TestLoadNodeGeometry:
    """Test node geometry loading."""

    def test_load_csv_geometry(self, sample_node_geometry_csv, sample_config):
        """Test loading geometry from CSV file."""
        positions, scales = load_node_geometry(sample_node_geometry_csv)

        assert positions.shape == (10, 2)
        assert scales.shape == (10,)
        assert np.all(positions[:, 0] >= 0) and np.all(positions[:, 0] <= np.pi)
        assert np.all(positions[:, 1] >= 0) and np.all(positions[:, 1] <= 2 * np.pi)
        assert np.all(scales > 0)
        assert np.allclose(scales, 300.0, rtol=0.1)

    def test_load_json_geometry(self, sample_node_geometry_json, sample_config):
        """Test loading geometry from JSON file."""
        positions, scales = load_node_geometry(sample_node_geometry_json)

        assert positions.shape == (10, 2)
        assert scales.shape == (10,)
        assert np.all(positions[:, 0] >= 0) and np.all(positions[:, 0] <= np.pi)
        assert np.all(positions[:, 1] >= 0) and np.all(positions[:, 1] <= 2 * np.pi)
        assert np.all(scales > 0)

    def test_load_geometry_degrees(self, temp_dir, sample_config):
        """Test loading geometry with degrees (should convert to radians)."""
        file_path = temp_dir / "node_geometry_deg.csv"

        n_nodes = 5
        theta_deg = np.linspace(10, 170, n_nodes)
        phi_deg = np.linspace(0, 360, n_nodes)
        scales = np.full(n_nodes, 300.0)

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["theta", "phi", "scale"])
            writer.writeheader()
            for i in range(n_nodes):
                writer.writerow(
                    {
                        "theta": theta_deg[i],
                        "phi": phi_deg[i],
                        "scale": scales[i],
                    }
                )

        positions, scales = load_node_geometry(file_path)

        assert positions.shape == (5, 2)
        # Check that values are in radians range
        assert np.all(positions[:, 0] <= np.pi)
        assert np.all(positions[:, 1] <= 2 * np.pi)

    def test_load_geometry_arcmin_scales(self, temp_dir, sample_config):
        """Test loading geometry with scales in arcmin (should convert to pc)."""
        file_path = temp_dir / "node_geometry_arcmin.csv"

        n_nodes = 5
        theta = np.linspace(0.1, np.pi - 0.1, n_nodes)
        phi = np.linspace(0, 2 * np.pi, n_nodes)
        scales_arcmin = np.full(n_nodes, 3.0)  # 3 arcmin

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["theta", "phi", "scale"])
            writer.writeheader()
            for i in range(n_nodes):
                writer.writerow(
                    {
                        "theta": theta[i],
                        "phi": phi[i],
                        "scale": scales_arcmin[i],
                    }
                )

        positions, scales = load_node_geometry(file_path)

        # Scales should be converted to pc (3 arcmin * 60 pc/arcmin = 180 pc)
        assert np.all(scales > 100)  # Should be in pc range

    def test_load_geometry_missing_file(self, sample_config):
        """Test loading geometry with missing file."""
        with pytest.raises(FileNotFoundError):
            load_node_geometry(Path("/nonexistent/file.csv"))

    def test_load_geometry_invalid_format(self, temp_dir, sample_config):
        """Test loading geometry with invalid format."""
        file_path = temp_dir / "invalid.txt"
        file_path.write_text("invalid data")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_node_geometry(file_path)

    def test_load_geometry_missing_columns(self, temp_dir, sample_config):
        """Test loading geometry with missing columns."""
        file_path = temp_dir / "incomplete.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["theta"])
            writer.writeheader()
            writer.writerow({"theta": 1.0})

        with pytest.raises(ValueError, match="phi"):
            load_node_geometry(file_path)

    def test_load_geometry_invalid_ranges(self, temp_dir, sample_config):
        """Test loading geometry with invalid position ranges."""
        file_path = temp_dir / "invalid_ranges.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["theta", "phi", "scale"])
            writer.writeheader()
            writer.writerow({"theta": 4.0, "phi": 1.0, "scale": 300.0})  # theta > π

        with pytest.raises(ValueError, match="Theta values must be in"):
            load_node_geometry(file_path)


class TestLoadNodeDepths:
    """Test node depth loading."""

    def test_load_csv_depths(self, sample_node_depths_csv, sample_config):
        """Test loading depths from CSV file."""
        depths = load_node_depths(sample_node_depths_csv)

        assert depths.shape == (10,)
        assert np.all(depths >= 0)
        assert np.all(depths >= 1e-5) and np.all(depths <= 1e-4)

    def test_load_json_depths(self, sample_node_depths_json, sample_config):
        """Test loading depths from JSON file."""
        depths = load_node_depths(sample_node_depths_json)

        assert depths.shape == (10,)
        assert np.all(depths >= 0)

    def test_load_depths_missing_file(self, sample_config):
        """Test loading depths with missing file."""
        with pytest.raises(FileNotFoundError):
            load_node_depths(Path("/nonexistent/file.csv"))

    def test_load_depths_negative(self, temp_dir, sample_config):
        """Test loading depths with negative values."""
        file_path = temp_dir / "negative_depths.csv"

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["depth"])
            writer.writeheader()
            writer.writerow({"depth": -0.1})

        with pytest.raises(ValueError, match="non-negative"):
            load_node_depths(file_path)


class TestMapDepthToTemperature:
    """Test depth to temperature mapping."""

    def test_map_depth_to_temperature(self, sample_config):
        """Test mapping depths to temperatures."""
        depths = np.array([1e-5, 1e-4, 1e-3])
        temperatures = map_depth_to_temperature(depths)

        assert temperatures.shape == (3,)
        assert np.all(temperatures >= 0)

        # Check formula: ΔT = (Δω/ω) * T_0 * 1e6 (μK)
        # For depth = 1e-5: ΔT = 1e-5 * 2.725 * 1e6 = 27.25 μK
        expected = 1e-5 * 2.725 * 1e6
        assert np.isclose(temperatures[0], expected, rtol=0.01)

    def test_map_depth_to_temperature_range(self, sample_config):
        """Test that temperatures are in expected range (20-30 μK)."""
        # Typical depths: 7e-6 to 1.1e-5 give ~20-30 μK
        depths = np.linspace(7e-6, 1.1e-5, 10)
        temperatures = map_depth_to_temperature(depths)

        assert np.all(temperatures >= 15)  # Allow some margin
        assert np.all(temperatures <= 35)  # Allow some margin

    def test_map_depth_to_temperature_negative(self, sample_config):
        """Test mapping with negative depths."""
        depths = np.array([-0.1, 0.1])

        with pytest.raises(ValueError, match="non-negative"):
            map_depth_to_temperature(depths)

    def test_map_depth_to_temperature_custom_config(self, sample_config):
        """Test mapping with custom config."""
        depths = np.array([1e-5])

        # Create custom config with different T_0
        custom_config = Config._load_defaults()
        custom_config.constants = custom_config.constants.__class__(
            D=custom_config.constants.D,
            z_CMB=custom_config.constants.z_CMB,
            T_0=3.0,  # Different T_0
            omega_CMB=custom_config.constants.omega_CMB,
            arcmin_to_pc_at_z1100=custom_config.constants.arcmin_to_pc_at_z1100,
            delta_T_min=custom_config.constants.delta_T_min,
            delta_T_max=custom_config.constants.delta_T_max,
        )

        temperatures = map_depth_to_temperature(depths, config=custom_config)

        # Should use custom T_0 = 3.0
        expected = 1e-5 * 3.0 * 1e6
        assert np.isclose(temperatures[0], expected, rtol=0.01)


class TestProcessNodeData:
    """Test complete node data processing."""

    def test_process_node_data(
        self,
        sample_node_geometry_csv,
        sample_node_depths_csv,
        sample_config,
    ):
        """Test processing complete node data."""
        node_data = process_node_data(
            geometry_path=sample_node_geometry_csv,
            depth_path=sample_node_depths_csv,
        )

        assert isinstance(node_data, ThetaNodeData)
        assert node_data.positions.shape == (10, 2)
        assert node_data.scales.shape == (10,)
        assert node_data.depths.shape == (10,)
        assert node_data.temperatures.shape == (10,)
        assert isinstance(node_data.metadata, dict)
        assert node_data.metadata["n_nodes"] == 10

    def test_process_node_data_inconsistent(
        self,
        sample_node_geometry_csv,
        temp_dir,
        sample_config,
    ):
        """Test processing with inconsistent data sizes."""
        # Create depths file with different number of nodes
        depth_path = temp_dir / "node_depths.csv"
        with open(depth_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["depth"])
            writer.writeheader()
            writer.writerow({"depth": 1e-5})  # Only 1 node

        with pytest.raises(ValueError, match="Inconsistent data"):
            process_node_data(
                geometry_path=sample_node_geometry_csv,
                depth_path=depth_path,
            )


class TestValidateNodeData:
    """Test node data validation."""

    def test_validate_node_data_valid(self, sample_config):
        """Test validation of valid node data."""
        n_nodes = 10
        positions = np.column_stack(
            [
                np.linspace(0.1, np.pi - 0.1, n_nodes),
                np.linspace(0, 2 * np.pi, n_nodes),
            ]
        )
        scales = np.full(n_nodes, 300.0)
        depths = np.linspace(7e-6, 1.1e-5, n_nodes)
        temperatures = map_depth_to_temperature(depths)
        metadata = {"n_nodes": n_nodes}

        node_data = ThetaNodeData(
            positions=positions,
            scales=scales,
            depths=depths,
            temperatures=temperatures,
            metadata=metadata,
        )

        assert validate_node_data(node_data) is True

    def test_validate_node_data_invalid_shape(self, sample_config):
        """Test validation with invalid array shapes."""
        n_nodes = 10
        positions = np.ones((n_nodes, 3))  # Wrong shape
        scales = np.full(n_nodes, 300.0)
        depths = np.linspace(7e-6, 1.1e-5, n_nodes)
        temperatures = map_depth_to_temperature(depths)
        metadata = {"n_nodes": n_nodes}

        node_data = ThetaNodeData(
            positions=positions,
            scales=scales,
            depths=depths,
            temperatures=temperatures,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="shape"):
            validate_node_data(node_data)

    def test_validate_node_data_invalid_positions(self, sample_config):
        """Test validation with invalid position ranges."""
        n_nodes = 10
        positions = np.column_stack(
            [
                np.full(n_nodes, 4.0),  # theta > π
                np.linspace(0, 2 * np.pi, n_nodes),
            ]
        )
        scales = np.full(n_nodes, 300.0)
        depths = np.linspace(7e-6, 1.1e-5, n_nodes)
        temperatures = map_depth_to_temperature(depths)
        metadata = {"n_nodes": n_nodes}

        node_data = ThetaNodeData(
            positions=positions,
            scales=scales,
            depths=depths,
            temperatures=temperatures,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="Theta values"):
            validate_node_data(node_data)

    def test_validate_node_data_negative_scales(self, sample_config):
        """Test validation with negative scales."""
        n_nodes = 10
        positions = np.column_stack(
            [
                np.linspace(0.1, np.pi - 0.1, n_nodes),
                np.linspace(0, 2 * np.pi, n_nodes),
            ]
        )
        scales = np.full(n_nodes, -100.0)  # Negative
        depths = np.linspace(7e-6, 1.1e-5, n_nodes)
        temperatures = map_depth_to_temperature(depths)
        metadata = {"n_nodes": n_nodes}

        node_data = ThetaNodeData(
            positions=positions,
            scales=scales,
            depths=depths,
            temperatures=temperatures,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="positive"):
            validate_node_data(node_data)

    def test_validate_node_data_negative_depths(self, sample_config):
        """Test validation with negative depths."""
        n_nodes = 10
        positions = np.column_stack(
            [
                np.linspace(0.1, np.pi - 0.1, n_nodes),
                np.linspace(0, 2 * np.pi, n_nodes),
            ]
        )
        scales = np.full(n_nodes, 300.0)
        depths = np.full(n_nodes, -0.1)  # Negative
        temperatures = map_depth_to_temperature(np.abs(depths))
        metadata = {"n_nodes": n_nodes}

        node_data = ThetaNodeData(
            positions=positions,
            scales=scales,
            depths=depths,
            temperatures=temperatures,
            metadata=metadata,
        )

        with pytest.raises(ValueError, match="non-negative"):
            validate_node_data(node_data)
