"""
Θ-field data loader for CMB verification project.

Loads frequency spectrum ρ_Θ(ω,t) and temporal evolution data
from data/theta/ directory.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional
import numpy as np
from dataclasses import dataclass
import logging
from config.settings import Config
from utils.io.data_loader import load_csv_data, load_json_data
from utils.io.data_index_loader import DataIndex
from cmb.theta_data_parser import (
    parse_csv_frequency_spectrum,
    parse_json_frequency_spectrum,
    parse_csv_evolution_data,
    parse_json_evolution_data,
)

# Try to import CUDA utilities for large array validation
try:
    from utils.cuda.array_model import CudaArray
    from utils.cuda.elementwise_vectorizer import ElementWiseVectorizer
    from utils.cuda.reduction_vectorizer import ReductionVectorizer

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    CudaArray = None  # type: ignore
    ElementWiseVectorizer = None  # type: ignore
    ReductionVectorizer = None  # type: ignore

logger = logging.getLogger(__name__)

# Threshold for using CUDA acceleration (array size)
CUDA_THRESHOLD = 1000000  # 1M elements


@dataclass
class ThetaFrequencySpectrum:
    """
    Θ-field frequency spectrum data.

    Attributes:
        frequencies: Frequency array ω in Hz
        times: Time array t
        spectrum: Spectrum values ρ_Θ(ω,t) as 2D array
        metadata: Additional metadata (units, ranges, etc.)
    """

    frequencies: np.ndarray
    times: np.ndarray
    spectrum: np.ndarray
    metadata: dict


@dataclass
class ThetaEvolution:
    """
    Θ-field temporal evolution data.

    Attributes:
        times: Time array t
        omega_min: ω_min(t) values
        omega_macro: ω_macro(t) values
        metadata: Additional metadata
    """

    times: np.ndarray
    omega_min: np.ndarray
    omega_macro: np.ndarray
    metadata: dict


def load_frequency_spectrum(
    data_path: Optional[Path] = None,
) -> ThetaFrequencySpectrum:
    """
    Load Θ-field frequency spectrum ρ_Θ(ω,t).

    Supports CSV and JSON formats. For CSV, expects columns:
    - 'frequency' or 'omega' or 'ω': frequency values
    - 'time' or 't': time values
    - 'spectrum' or 'rho' or 'ρ': spectrum values (2D array)

    For JSON, expects structure:
    {
        "frequencies": [...],
        "times": [...],
        "spectrum": [[...], [...], ...],
        "metadata": {...}
    }

    Args:
        data_path: Path to frequency spectrum data file.
                   If None, uses path from data index.

    Returns:
        ThetaFrequencySpectrum instance

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    # Get data path from index if not provided
    if data_path is None:
        config = Config._load_defaults()
        data_index = DataIndex.load(config.paths.data_index)

        # Search for frequency spectrum file in theta_field_data category
        theta_files = data_index.get_files_by_category("theta_field_data")
        spectrum_file = None

        for file_info in theta_files:
            file_name = file_info.get("name", "")
            if any(
                keyword in file_name.lower()
                for keyword in ["frequency", "spectrum", "rho", "omega"]
            ):
                spectrum_file = data_index.get_file_path(  # noqa: E501
                    "theta_field_data", file_name
                )
                if spectrum_file and spectrum_file.exists():
                    break

        if spectrum_file is None or not spectrum_file.exists():
            # Try direct path in data/theta/
            spectrum_file = config.paths.data_theta / "frequency_spectrum.csv"
            if not spectrum_file.exists():
                spectrum_file = (  # noqa: E501
                    config.paths.data_theta / "frequency_spectrum.json"
                )
                if not spectrum_file.exists():
                    raise FileNotFoundError(
                        f"Frequency spectrum file not found in data/theta/ "
                        f"or data index. Searched: {config.paths.data_theta}"
                    )

        data_path = spectrum_file

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(  # noqa: E501
            f"Frequency spectrum file not found: {data_path}"
        )

    # Determine file format and load
    if data_path.suffix.lower() == ".json":
        data = load_json_data(data_path)
        return parse_json_frequency_spectrum(data, data_path)
    else:
        # Assume CSV format
        data = load_csv_data(data_path)
        return parse_csv_frequency_spectrum(data, data_path)


def load_evolution_data(data_path: Optional[Path] = None) -> ThetaEvolution:
    """
    Load Θ-field temporal evolution data (ω_min(t), ω_macro(t)).

    Supports CSV and JSON formats. For CSV, expects columns:
    - 'time' or 't': time values
    - 'omega_min' or 'ω_min': ω_min(t) values
    - 'omega_macro' or 'ω_macro': ω_macro(t) values

    For JSON, expects structure:
    {
        "times": [...],
        "omega_min": [...],
        "omega_macro": [...],
        "metadata": {...}
    }

    Args:
        data_path: Path to evolution data file.
                   If None, uses path from data index.

    Returns:
        ThetaEvolution instance

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    # Get data path from index if not provided
    if data_path is None:
        config = Config._load_defaults()
        data_index = DataIndex.load(config.paths.data_index)

        # Search for evolution file in theta_field_data category
        theta_files = data_index.get_files_by_category("theta_field_data")
        evolution_file = None

        for file_info in theta_files:
            file_name = file_info.get("name", "")
            if any(
                keyword in file_name.lower()
                for keyword in ["evolution", "omega_min", "omega_macro"]
            ):
                evolution_file = data_index.get_file_path(  # noqa: E501
                    "theta_field_data", file_name
                )
                if evolution_file and evolution_file.exists():
                    break

        if evolution_file is None or not evolution_file.exists():
            # Try direct path in data/theta/
            evolution_file = config.paths.data_theta / "evolution.csv"
            if not evolution_file.exists():
                evolution_file = config.paths.data_theta / "evolution.json"
                if not evolution_file.exists():
                    raise FileNotFoundError(
                        f"Evolution data file not found in data/theta/ "
                        f"or data index. Searched: {config.paths.data_theta}"
                    )

        data_path = evolution_file

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Evolution data file not found: {data_path}")

    # Determine file format and load
    if data_path.suffix.lower() == ".json":
        data = load_json_data(data_path)
        return parse_json_evolution_data(data, data_path)
    else:
        # Assume CSV format
        data = load_csv_data(data_path)
        return parse_csv_evolution_data(data, data_path)


def validate_frequency_spectrum(spectrum: ThetaFrequencySpectrum) -> bool:
    """
    Validate frequency spectrum data.

    Checks:
    - Non-empty arrays
    - Positive frequencies
    - Valid time range
    - Non-negative spectrum values
    - Consistent array shapes (spectrum shape = (len(frequencies), len(times)))

    Args:
        spectrum: ThetaFrequencySpectrum to validate

    Returns:
        True if valid

    Raises:
        ValueError: If data is invalid
    """
    # Check non-empty arrays
    if spectrum.frequencies.size == 0:
        raise ValueError("Frequency array is empty")
    if spectrum.times.size == 0:
        raise ValueError("Time array is empty")
    if spectrum.spectrum.size == 0:
        raise ValueError("Spectrum array is empty")

    # Check positive frequencies (use CUDA for large arrays)
    if (
        CUDA_AVAILABLE
        and spectrum.frequencies.size > CUDA_THRESHOLD
        and CudaArray is not None
        and ElementWiseVectorizer is not None
        and ReductionVectorizer is not None
    ):
        # Use CUDA-accelerated validation for large arrays
        cuda_freq = CudaArray(spectrum.frequencies, device="cuda")
        vectorizer = ElementWiseVectorizer(use_gpu=True)
        reducer = ReductionVectorizer(use_gpu=True)

        # Check for non-positive values on GPU
        non_positive = vectorizer.vectorize_operation(
            cuda_freq, "less_equal", 0.0
        )
        has_non_positive = reducer.vectorize_reduction(non_positive, "any")
        if has_non_positive:
            count_result = reducer.vectorize_reduction(non_positive, "sum")
            # Convert to int (handles CudaArray, float, int)
            if hasattr(count_result, "to_numpy"):
                count = int(count_result.to_numpy().item())
            else:
                count = int(count_result)
            raise ValueError(
                "All frequencies must be positive. "
                f"Found non-positive values: {count}"
            )
    else:
        # CPU path
        if np.any(spectrum.frequencies <= 0):
            raise ValueError(
                "All frequencies must be positive. "
                f"Found non-positive values: "
                f"{np.sum(spectrum.frequencies <= 0)}"
            )

    # Check non-negative spectrum values (use CUDA for large arrays)
    if (
        CUDA_AVAILABLE
        and spectrum.spectrum.size > CUDA_THRESHOLD
        and CudaArray is not None
        and ElementWiseVectorizer is not None
        and ReductionVectorizer is not None
    ):
        # Use CUDA-accelerated validation for large arrays
        cuda_spec = CudaArray(spectrum.spectrum, device="cuda")
        vectorizer = ElementWiseVectorizer(use_gpu=True)
        reducer = ReductionVectorizer(use_gpu=True)

        # Check for negative values on GPU
        negative = vectorizer.vectorize_operation(cuda_spec, "less", 0.0)
        has_negative = reducer.vectorize_reduction(negative, "any")
        if has_negative:
            count_result = reducer.vectorize_reduction(negative, "sum")
            # Convert to int (handles CudaArray, float, int)
            if hasattr(count_result, "to_numpy"):
                count = int(count_result.to_numpy().item())
            else:
                count = int(count_result)
            raise ValueError(
                "All spectrum values must be non-negative. "
                f"Found negative values: {count}"
            )
    else:
        # CPU path
        if np.any(spectrum.spectrum < 0):
            raise ValueError(
                "All spectrum values must be non-negative. "
                f"Found negative values: {np.sum(spectrum.spectrum < 0)}"
            )

    # Check consistent array shapes
    expected_shape = (len(spectrum.frequencies), len(spectrum.times))
    if spectrum.spectrum.shape != expected_shape:
        raise ValueError(
            f"Spectrum shape mismatch: expected {expected_shape} "
            f"(frequencies × times), got {spectrum.spectrum.shape}"
        )

    # Check for NaN or Inf values (use CUDA for large arrays)
    if (
        CUDA_AVAILABLE
        and spectrum.frequencies.size > CUDA_THRESHOLD
        and CudaArray is not None
        and ElementWiseVectorizer is not None
        and ReductionVectorizer is not None
    ):
        # Use CUDA-accelerated validation for large arrays
        cuda_freq = CudaArray(spectrum.frequencies, device="cuda")
        cuda_times = CudaArray(spectrum.times, device="cuda")
        cuda_spec = CudaArray(spectrum.spectrum, device="cuda")
        reducer = ReductionVectorizer(use_gpu=True)

        # Check NaN on GPU (convert to numpy for isnan check)
        if reducer.vectorize_reduction(
            CudaArray(np.isnan(cuda_freq.to_numpy()), device="cuda"), "any"
        ):
            raise ValueError("Frequency array contains NaN values")
        if reducer.vectorize_reduction(
            CudaArray(np.isnan(cuda_times.to_numpy()), device="cuda"), "any"
        ):
            raise ValueError("Time array contains NaN values")
        if reducer.vectorize_reduction(
            CudaArray(np.isnan(cuda_spec.to_numpy()), device="cuda"), "any"
        ):
            raise ValueError("Spectrum array contains NaN values")

        # Check Inf on GPU
        if reducer.vectorize_reduction(
            CudaArray(np.isinf(cuda_freq.to_numpy()), device="cuda"), "any"
        ):
            raise ValueError("Frequency array contains Inf values")
        if reducer.vectorize_reduction(
            CudaArray(np.isinf(cuda_times.to_numpy()), device="cuda"), "any"
        ):
            raise ValueError("Time array contains Inf values")
        if reducer.vectorize_reduction(
            CudaArray(np.isinf(cuda_spec.to_numpy()), device="cuda"), "any"
        ):
            raise ValueError("Spectrum array contains Inf values")
    else:
        # CPU path
        if np.any(np.isnan(spectrum.frequencies)):
            raise ValueError("Frequency array contains NaN values")
        if np.any(np.isnan(spectrum.times)):
            raise ValueError("Time array contains NaN values")
        if np.any(np.isnan(spectrum.spectrum)):
            raise ValueError("Spectrum array contains NaN values")
        if np.any(np.isinf(spectrum.frequencies)):
            raise ValueError("Frequency array contains Inf values")
        if np.any(np.isinf(spectrum.times)):
            raise ValueError("Time array contains Inf values")
        if np.any(np.isinf(spectrum.spectrum)):
            raise ValueError("Spectrum array contains Inf values")

    return True


def validate_evolution_data(evolution: ThetaEvolution) -> bool:
    """
    Validate evolution data.

    Checks:
    - Non-empty arrays
    - Valid time range
    - Positive omega values
    - Consistent array lengths (all arrays must have same length)

    Args:
        evolution: ThetaEvolution to validate

    Returns:
        True if valid

    Raises:
        ValueError: If data is invalid
    """
    # Check non-empty arrays
    if evolution.times.size == 0:
        raise ValueError("Time array is empty")
    if evolution.omega_min.size == 0:
        raise ValueError("omega_min array is empty")
    if evolution.omega_macro.size == 0:
        raise ValueError("omega_macro array is empty")

    # Check consistent array lengths
    n_times = len(evolution.times)
    if len(evolution.omega_min) != n_times:
        raise ValueError(
            f"Array length mismatch: times has {n_times} elements, "
            f"omega_min has {len(evolution.omega_min)} elements"
        )
    if len(evolution.omega_macro) != n_times:
        raise ValueError(
            f"Array length mismatch: times has {n_times} elements, "
            f"omega_macro has {len(evolution.omega_macro)} elements"
        )

    # Check positive omega values (use CUDA for large arrays)
    if (
        CUDA_AVAILABLE
        and evolution.omega_min.size > CUDA_THRESHOLD
        and CudaArray is not None
        and ElementWiseVectorizer is not None
        and ReductionVectorizer is not None
    ):
        # Use CUDA-accelerated validation for large arrays
        cuda_omega_min = CudaArray(evolution.omega_min, device="cuda")
        cuda_omega_macro = CudaArray(evolution.omega_macro, device="cuda")
        vectorizer = ElementWiseVectorizer(use_gpu=True)
        reducer = ReductionVectorizer(use_gpu=True)

        # Check for non-positive values on GPU
        non_positive_min = vectorizer.vectorize_operation(
            cuda_omega_min, "less_equal", 0.0
        )
        has_non_positive_min = reducer.vectorize_reduction(
            non_positive_min, "any"
        )
        if has_non_positive_min:
            count_result = reducer.vectorize_reduction(non_positive_min, "sum")
            # Convert to int (handles CudaArray, float, int)
            if hasattr(count_result, "to_numpy"):
                count = int(count_result.to_numpy().item())
            else:
                count = int(count_result)
            raise ValueError(
                "All omega_min values must be positive. "
                f"Found non-positive values: {count}"
            )

        non_positive_macro = vectorizer.vectorize_operation(
            cuda_omega_macro, "less_equal", 0.0
        )
        has_non_positive_macro = reducer.vectorize_reduction(
            non_positive_macro, "any"
        )
        if has_non_positive_macro:
            count_result = reducer.vectorize_reduction(
                non_positive_macro, "sum"
            )
            # Convert to int (handles CudaArray, float, int)
            if hasattr(count_result, "to_numpy"):
                count = int(count_result.to_numpy().item())
            else:
                count = int(count_result)
            raise ValueError(
                "All omega_macro values must be positive. "
                f"Found non-positive values: {count}"
            )
    else:
        # CPU path
        if np.any(evolution.omega_min <= 0):
            raise ValueError(
                "All omega_min values must be positive. "
                f"Found non-positive values: "
                f"{np.sum(evolution.omega_min <= 0)}"
            )
        if np.any(evolution.omega_macro <= 0):
            raise ValueError(
                "All omega_macro values must be positive. "
                f"Found non-positive values: "
                f"{np.sum(evolution.omega_macro <= 0)}"
            )

    # Check for NaN or Inf values
    if np.any(np.isnan(evolution.times)):
        raise ValueError("Time array contains NaN values")
    if np.any(np.isnan(evolution.omega_min)):
        raise ValueError("omega_min array contains NaN values")
    if np.any(np.isnan(evolution.omega_macro)):
        raise ValueError("omega_macro array contains NaN values")
    if np.any(np.isinf(evolution.times)):
        raise ValueError("Time array contains Inf values")
    if np.any(np.isinf(evolution.omega_min)):
        raise ValueError("omega_min array contains Inf values")
    if np.any(np.isinf(evolution.omega_macro)):
        raise ValueError("omega_macro array contains Inf values")

    # Check that omega_min < omega_macro (physical constraint)
    # Use CUDA for large arrays
    if (
        CUDA_AVAILABLE
        and evolution.omega_min.size > CUDA_THRESHOLD
        and CudaArray is not None
        and ElementWiseVectorizer is not None
        and ReductionVectorizer is not None
    ):
        # Use CUDA-accelerated validation for large arrays
        cuda_omega_min = CudaArray(evolution.omega_min, device="cuda")
        cuda_omega_macro = CudaArray(evolution.omega_macro, device="cuda")
        vectorizer = ElementWiseVectorizer(use_gpu=True)
        reducer = ReductionVectorizer(use_gpu=True)

        # Check omega_min >= omega_macro on GPU
        violation = vectorizer.vectorize_operation(
            cuda_omega_min, "greater_equal", cuda_omega_macro
        )
        has_violation = reducer.vectorize_reduction(violation, "any")
        if has_violation:
            count_result = reducer.vectorize_reduction(violation, "sum")
            # Convert to int (handles CudaArray, float, int)
            if hasattr(count_result, "to_numpy"):
                count = int(count_result.to_numpy().item())
            else:
                count = int(count_result)
            raise ValueError(
                f"Physical constraint violated: "
                f"omega_min must be < omega_macro. "
                f"Found {count} invalid value(s)"
            )
    else:
        # CPU path
        if np.any(evolution.omega_min >= evolution.omega_macro):
            invalid_count = np.sum(
                evolution.omega_min >= evolution.omega_macro
            )
            raise ValueError(
                f"Physical constraint violated: "
                f"omega_min must be < omega_macro. "
                f"Found {invalid_count} invalid value(s)"
            )

    return True
