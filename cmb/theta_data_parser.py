"""
Θ-field data parsers for CMB verification project.

Provides parsing functions for CSV and JSON data formats.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from cmb.theta_data_loader import ThetaFrequencySpectrum, ThetaEvolution


def parse_csv_frequency_spectrum(
    data: Dict[str, np.ndarray],
    file_path: Path,
    block_size: Optional[int] = None,
) -> "ThetaFrequencySpectrum":  # noqa: F821
    """
    Parse frequency spectrum from CSV data.

    Supports block processing for large datasets to reduce memory usage.

    Args:
        data: Dictionary with CSV column data
        file_path: Path to source file (for error messages)
        block_size: Optional block size for processing large arrays.
                   If None, processes entire array at once.

    Returns:
        ThetaFrequencySpectrum instance

    Raises:
        ValueError: If required columns are missing or data format is invalid
    """
    # Find frequency column
    freq_key = None
    for key in ["frequency", "omega", "ω", "f", "freq"]:
        if key in data:
            freq_key = key
            break

    if freq_key is None:
        raise ValueError(
            f"Frequency column not found in CSV file {file_path}. "
            f"Expected one of: frequency, omega, ω, f, freq. "
            f"Found columns: {list(data.keys())}"
        )

    # Find time column
    time_key = None
    for key in ["time", "t"]:
        if key in data:
            time_key = key
            break

    if time_key is None:
        raise ValueError(
            f"Time column not found in CSV file {file_path}. "
            f"Expected one of: time, t. "
            f"Found columns: {list(data.keys())}"
        )

    # Find spectrum column
    spectrum_key = None
    for key in ["spectrum", "rho", "ρ", "rho_theta"]:
        if key in data:
            spectrum_key = key
            break

    if spectrum_key is None:
        raise ValueError(
            f"Spectrum column not found in CSV file {file_path}. "
            f"Expected one of: spectrum, rho, ρ, rho_theta. "
            f"Found columns: {list(data.keys())}"
        )

    frequencies_raw = np.asarray(data[freq_key], dtype=np.float64)
    times_raw = np.asarray(data[time_key], dtype=np.float64)
    spectrum_data = np.asarray(data[spectrum_key], dtype=np.float64)

    # Extract unique frequencies and times (in case CSV has
    # repeated values for each combination)
    frequencies = np.unique(frequencies_raw)
    times = np.unique(times_raw)

    # Handle 2D spectrum: if spectrum is 1D, reshape based on
    # frequencies and times. For CSV, spectrum might be stored as
    # flattened array
    if spectrum_data.ndim == 1:
        # Check if we have the right number of values
        expected_size = len(frequencies) * len(times)
        if spectrum_data.size == expected_size:
            # Reshape: assume frequencies × times
            spectrum = spectrum_data.reshape(len(frequencies), len(times))
        elif spectrum_data.size == len(frequencies_raw):
            # CSV format: one value per row, need to reshape based on
            # original frequencies and times order
            # Vectorized approach using numpy searchsorted
            spectrum_2d = np.zeros((len(frequencies), len(times)))

            # Use searchsorted for vectorized index finding
            # This is faster than dictionary lookup for large arrays
            freq_indices = np.searchsorted(frequencies, frequencies_raw)
            time_indices = np.searchsorted(times, times_raw)

            # Block processing for very large arrays to reduce memory usage
            if block_size is not None and len(frequencies_raw) > block_size:
                # Process in blocks
                n_blocks = (
                    len(frequencies_raw) + block_size - 1
                ) // block_size
                for i in range(n_blocks):
                    start_idx = i * block_size
                    end_idx = min((i + 1) * block_size, len(frequencies_raw))
                    block_freq_indices = freq_indices[start_idx:end_idx]
                    block_time_indices = time_indices[start_idx:end_idx]
                    block_spectrum = spectrum_data[start_idx:end_idx]
                    # Vectorized assignment for block
                    spectrum_2d[
                        block_freq_indices, block_time_indices
                    ] = block_spectrum
            else:
                # Vectorized assignment using advanced indexing (entire array)
                spectrum_2d[freq_indices, time_indices] = spectrum_data

            spectrum = spectrum_2d
        else:
            raise ValueError(
                f"Cannot reshape spectrum array: size {spectrum_data.size} "
                f"does not match expected {expected_size} "
                f"(frequencies={len(frequencies)}, times={len(times)})"
            )
    elif spectrum_data.ndim == 2:
        spectrum = spectrum_data
    else:
        raise ValueError(
            f"Spectrum must be 1D or 2D array, got {spectrum_data.ndim}D"
        )

    # Extract metadata
    metadata = {
        "source_file": str(file_path),
        "format": "CSV",
        "frequency_range": (
            float(np.min(frequencies)),
            float(np.max(frequencies)),
        ),
        "time_range": (float(np.min(times)), float(np.max(times))),
        "n_frequencies": len(frequencies),
        "n_times": len(times),
    }

    from cmb.theta_data_loader import ThetaFrequencySpectrum

    return ThetaFrequencySpectrum(
        frequencies=frequencies,
        times=times,
        spectrum=spectrum,
        metadata=metadata,
    )


def parse_json_frequency_spectrum(
    data: Dict[str, Any], file_path: Path
) -> "ThetaFrequencySpectrum":  # noqa: F821
    """
    Parse frequency spectrum from JSON data.

    Args:
        data: Dictionary with JSON data
        file_path: Path to source file (for error messages)

    Returns:
        ThetaFrequencySpectrum instance

    Raises:
        ValueError: If required keys are missing or data format is invalid
    """
    # Extract required arrays
    if "frequencies" not in data:
        raise ValueError(
            f"'frequencies' key not found in JSON file {file_path}. "
            f"Found keys: {list(data.keys())}"
        )
    if "times" not in data:
        raise ValueError(
            f"'times' key not found in JSON file {file_path}. "
            f"Found keys: {list(data.keys())}"
        )
    if "spectrum" not in data:
        raise ValueError(
            f"'spectrum' key not found in JSON file {file_path}. "
            f"Found keys: {list(data.keys())}"
        )

    frequencies = np.asarray(data["frequencies"], dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64)
    spectrum = np.asarray(data["spectrum"], dtype=np.float64)

    # Ensure spectrum is 2D
    if spectrum.ndim == 1:
        expected_size = len(frequencies) * len(times)
        if spectrum.size == expected_size:
            spectrum = spectrum.reshape(len(frequencies), len(times))
        else:
            raise ValueError(
                f"Cannot reshape spectrum array: size {spectrum.size} "
                f"does not match expected {expected_size}"
            )
    elif spectrum.ndim != 2:
        raise ValueError(
            f"Spectrum must be 1D or 2D array, got {spectrum.ndim}D"
        )

    # Extract metadata
    metadata = data.get("metadata", {})
    metadata["source_file"] = str(file_path)
    metadata["format"] = "JSON"
    if "frequency_range" not in metadata:
        metadata["frequency_range"] = (
            float(np.min(frequencies)),
            float(np.max(frequencies)),
        )
    if "time_range" not in metadata:
        metadata["time_range"] = (float(np.min(times)), float(np.max(times)))
    if "n_frequencies" not in metadata:
        metadata["n_frequencies"] = len(frequencies)
    if "n_times" not in metadata:
        metadata["n_times"] = len(times)  # noqa: E501

    from cmb.theta_data_loader import ThetaFrequencySpectrum

    return ThetaFrequencySpectrum(
        frequencies=frequencies,
        times=times,
        spectrum=spectrum,
        metadata=metadata,
    )


def parse_csv_evolution_data(
    data: Dict[str, np.ndarray], file_path: Path
) -> "ThetaEvolution":  # noqa: F821
    """
    Parse evolution data from CSV data.

    Args:
        data: Dictionary with CSV column data
        file_path: Path to source file (for error messages)

    Returns:
        ThetaEvolution instance

    Raises:
        ValueError: If required columns are missing or data format is invalid
    """
    # Find time column
    time_key = None
    for key in ["time", "t"]:
        if key in data:
            time_key = key
            break

    if time_key is None:
        raise ValueError(
            f"Time column not found in CSV file {file_path}. "
            f"Expected one of: time, t. "
            f"Found columns: {list(data.keys())}"
        )

    # Find omega_min column
    omega_min_key = None
    for key in ["omega_min", "ω_min", "omega_min_t", "ω_min_t"]:
        if key in data:
            omega_min_key = key
            break

    if omega_min_key is None:
        raise ValueError(
            f"omega_min column not found in CSV file {file_path}. "
            f"Expected one of: omega_min, ω_min, "
            f"omega_min_t, ω_min_t. "
            f"Found columns: {list(data.keys())}"
        )

    # Find omega_macro column
    omega_macro_key = None
    for key in ["omega_macro", "ω_macro", "omega_macro_t", "ω_macro_t"]:
        if key in data:
            omega_macro_key = key
            break

    if omega_macro_key is None:
        raise ValueError(
            f"omega_macro column not found in CSV file {file_path}. "
            f"Expected one of: omega_macro, ω_macro, "
            f"omega_macro_t, ω_macro_t. "
            f"Found columns: {list(data.keys())}"
        )

    times = np.asarray(data[time_key], dtype=np.float64)
    omega_min = np.asarray(data[omega_min_key], dtype=np.float64)
    omega_macro = np.asarray(data[omega_macro_key], dtype=np.float64)

    # Extract metadata
    metadata = {
        "source_file": str(file_path),
        "format": "CSV",
        "time_range": (float(np.min(times)), float(np.max(times))),
        "n_points": len(times),
    }

    from cmb.theta_data_loader import ThetaEvolution

    return ThetaEvolution(
        times=times,
        omega_min=omega_min,
        omega_macro=omega_macro,
        metadata=metadata,
    )


def parse_json_evolution_data(
    data: Dict[str, Any], file_path: Path
) -> "ThetaEvolution":  # noqa: F821
    """
    Parse evolution data from JSON data.

    Args:
        data: Dictionary with JSON data
        file_path: Path to source file (for error messages)

    Returns:
        ThetaEvolution instance

    Raises:
        ValueError: If required keys are missing or data format is invalid
    """
    # Extract required arrays
    if "times" not in data:
        raise ValueError(
            f"'times' key not found in JSON file {file_path}. "
            f"Found keys: {list(data.keys())}"
        )
    if "omega_min" not in data:
        raise ValueError(
            f"'omega_min' key not found in JSON file {file_path}. "
            f"Found keys: {list(data.keys())}"
        )
    if "omega_macro" not in data:
        raise ValueError(
            f"'omega_macro' key not found in JSON file {file_path}. "
            f"Found keys: {list(data.keys())}"
        )

    times = np.asarray(data["times"], dtype=np.float64)
    omega_min = np.asarray(data["omega_min"], dtype=np.float64)
    omega_macro = np.asarray(data["omega_macro"], dtype=np.float64)

    # Extract metadata
    metadata = data.get("metadata", {})
    metadata["source_file"] = str(file_path)
    metadata["format"] = "JSON"
    if "time_range" not in metadata:
        metadata["time_range"] = (float(np.min(times)), float(np.max(times)))
    if "n_points" not in metadata:
        metadata["n_points"] = len(times)

    from cmb.theta_data_loader import ThetaEvolution

    return ThetaEvolution(
        times=times,
        omega_min=omega_min,
        omega_macro=omega_macro,
        metadata=metadata,
    )
