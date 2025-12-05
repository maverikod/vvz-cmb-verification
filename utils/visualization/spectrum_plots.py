"""
Power spectrum visualization utilities.

Provides functions for plotting C_l power spectra and comparisons.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_power_spectrum(
    multipoles: np.ndarray,
    spectrum: np.ndarray,
    errors: Optional[np.ndarray] = None,
    title: str = "Power Spectrum",
    output_path: Optional[Path] = None,
    log_scale: bool = True,
) -> None:
    """
    Plot C_l power spectrum.

    Args:
        multipoles: Multipole array l
        spectrum: Power spectrum values C_l
        errors: Error bars (optional)
        title: Plot title
        output_path: Path to save plot (optional)
        log_scale: Whether to use logarithmic scale

    Raises:
        ValueError: If data arrays are invalid
    """
    if not isinstance(multipoles, np.ndarray) or not isinstance(spectrum, np.ndarray):
        raise ValueError("Multipoles and spectrum must be numpy arrays")

    if multipoles.size != spectrum.size:
        raise ValueError(
            f"Array sizes must match: {multipoles.size} vs {spectrum.size}"
        )

    if errors is not None and errors.size != spectrum.size:
        raise ValueError(
            f"Errors size must match spectrum: {errors.size} vs {spectrum.size}"
        )

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if errors is not None:
            ax.errorbar(
                multipoles,
                spectrum,
                yerr=errors,
                fmt="o",
                markersize=3,
                capsize=2,
                label="Power Spectrum",
            )
        else:
            ax.plot(multipoles, spectrum, "b-", linewidth=1.5, label="Power Spectrum")

        ax.set_xlabel("Multipole l", fontsize=12)
        ax.set_ylabel("C_l [μK²]", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if log_scale:
            ax.set_yscale("log")
            ax.set_xscale("log")

        plt.tight_layout()

        # Save if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        raise ValueError(f"Failed to plot power spectrum: {e}") from e


def plot_spectrum_comparison(
    multipoles: np.ndarray,
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    label1: str = "Reconstructed",
    label2: str = "Observed",
    title: str = "Spectrum Comparison",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot comparison of two power spectra.

    Args:
        multipoles: Multipole array l
        spectrum1: First power spectrum
        spectrum2: Second power spectrum
        label1: Label for first spectrum
        label2: Label for second spectrum
        title: Plot title
        output_path: Path to save plot (optional)
    """
    if not isinstance(multipoles, np.ndarray):
        raise ValueError("Multipoles must be a numpy array")

    if not isinstance(spectrum1, np.ndarray) or not isinstance(spectrum2, np.ndarray):
        raise ValueError("Both spectra must be numpy arrays")

    if multipoles.size != spectrum1.size or multipoles.size != spectrum2.size:
        raise ValueError("All arrays must have the same size")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(multipoles, spectrum1, "b-", linewidth=1.5, label=label1)
        ax.plot(multipoles, spectrum2, "r--", linewidth=1.5, label=label2)

        ax.set_xlabel("Multipole l", fontsize=12)
        ax.set_ylabel("C_l [μK²]", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        raise ValueError(f"Failed to plot spectrum comparison: {e}") from e


def plot_high_l_peak(
    multipoles: np.ndarray,
    spectrum: np.ndarray,
    peak_range: Tuple[float, float],
    title: str = "High-l Peak",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot high-l peak region with highlighting.

    Args:
        multipoles: Multipole array l
        spectrum: Power spectrum values
        peak_range: (l_min, l_max) for peak region
        title: Plot title
        output_path: Path to save plot (optional)
    """
    if not isinstance(multipoles, np.ndarray) or not isinstance(spectrum, np.ndarray):
        raise ValueError("Multipoles and spectrum must be numpy arrays")

    if multipoles.size != spectrum.size:
        raise ValueError("Array sizes must match")

    l_min, l_max = peak_range
    if l_min >= l_max:
        raise ValueError(f"l_min ({l_min}) must be less than l_max ({l_max})")

    try:
        # Filter data to peak range
        mask = (multipoles >= l_min) & (multipoles <= l_max)
        l_filtered = multipoles[mask]
        cl_filtered = spectrum[mask]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot full spectrum in light gray
        ax.plot(
            multipoles,
            spectrum,
            "lightgray",
            linewidth=1,
            alpha=0.5,
            label="Full Spectrum",
        )

        # Highlight peak region
        ax.plot(l_filtered, cl_filtered, "b-", linewidth=2, label="Peak Region")
        ax.axvline(l_min, color="r", linestyle="--", alpha=0.7, label="Range")
        ax.axvline(l_max, color="r", linestyle="--", alpha=0.7)

        ax.set_xlabel("Multipole l", fontsize=12)
        ax.set_ylabel("C_l [μK²]", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        raise ValueError(f"Failed to plot high-l peak: {e}") from e


def plot_subpeaks(
    multipoles: np.ndarray,
    spectrum: np.ndarray,
    peak_positions: List[float],
    title: str = "Sub-peaks",
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot power spectrum with sub-peaks highlighted.

    Args:
        multipoles: Multipole array l
        spectrum: Power spectrum values
        peak_positions: List of peak l positions
        title: Plot title
        output_path: Path to save plot (optional)
    """
    if not isinstance(multipoles, np.ndarray) or not isinstance(spectrum, np.ndarray):
        raise ValueError("Multipoles and spectrum must be numpy arrays")

    if multipoles.size != spectrum.size:
        raise ValueError("Array sizes must match")

    if not peak_positions:
        raise ValueError("peak_positions list cannot be empty")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot spectrum
        ax.plot(multipoles, spectrum, "b-", linewidth=1.5, label="Power Spectrum")

        # Mark peak positions
        for peak_l in peak_positions:
            # Find closest multipole
            idx = np.argmin(np.abs(multipoles - peak_l))
            peak_cl = spectrum[idx]
            ax.plot(
                peak_l,
                peak_cl,
                "ro",
                markersize=8,
                label="Peak" if peak_l == peak_positions[0] else "",
            )
            ax.annotate(
                f"l={peak_l:.0f}",
                xy=(peak_l, peak_cl),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_xlabel("Multipole l", fontsize=12)
        ax.set_ylabel("C_l [μK²]", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save if path provided
        if output_path is not None:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        raise ValueError(f"Failed to plot sub-peaks: {e}") from e


