"""
Memory safety utilities for CUDA operations.

Provides additional safety checks and emergency cleanup to prevent system freeze.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import threading
import time
from typing import Optional, Callable, Dict
from contextlib import contextmanager

from utils.cuda.exceptions import CudaMemoryError

# Try to import CuPy
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class MemorySafetyGuard:
    """
    Safety guard for CUDA memory operations.

    Provides timeout protection and emergency cleanup to prevent system freeze.
    """

    def __init__(self, timeout_seconds: float = 30.0):
        """
        Initialize safety guard.

        Args:
            timeout_seconds: Maximum time for operation before emergency cleanup
        """
        self.timeout_seconds = timeout_seconds
        self._active_guards: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    @contextmanager
    def protect_operation(
        self, operation_name: str, cleanup_func: Optional[Callable] = None
    ):
        """
        Protect a CUDA operation with timeout and emergency cleanup.

        Args:
            operation_name: Name of the operation
            cleanup_func: Optional cleanup function to call on timeout

        Yields:
            None

        Raises:
            CudaMemoryError: If operation times out
        """
        operation_id = f"{operation_name}_{id(self)}"
        timeout_occurred = threading.Event()

        def timeout_handler():
            """Handle timeout."""
            time.sleep(self.timeout_seconds)
            if not timeout_occurred.is_set():
                timeout_occurred.set()
                # Emergency cleanup
                if cleanup_func:
                    try:
                        cleanup_func()
                    except Exception:
                        pass
                # Force GPU memory cleanup
                if CUPY_AVAILABLE:
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass

        # Start timeout thread
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()

        with self._lock:
            self._active_guards[operation_id] = timeout_thread

        try:
            yield
            # Check if timeout occurred
            if timeout_occurred.is_set():
                raise CudaMemoryError(
                    f"Operation '{operation_name}' timed out after "
                    f"{self.timeout_seconds} seconds - emergency cleanup performed"
                )
        finally:
            timeout_occurred.set()
            with self._lock:
                if operation_id in self._active_guards:
                    del self._active_guards[operation_id]

    def emergency_cleanup(self) -> None:
        """
        Perform emergency GPU memory cleanup.

        CRITICAL: Called when system is about to freeze.
        """
        if not CUPY_AVAILABLE:
            return

        try:
            # Force free all GPU memory
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            cp.get_default_memory_pool().free_all_blocks()

            # Clear all active guards
            with self._lock:
                self._active_guards.clear()
        except Exception:
            # Ignore errors during emergency cleanup
            pass


# Global safety guard instance
_safety_guard: Optional[MemorySafetyGuard] = None
_safety_guard_lock = threading.Lock()


def get_safety_guard() -> MemorySafetyGuard:
    """
    Get global safety guard instance.

    Returns:
        MemorySafetyGuard instance
    """
    global _safety_guard
    if _safety_guard is None:
        with _safety_guard_lock:
            if _safety_guard is None:
                _safety_guard = MemorySafetyGuard()
    return _safety_guard


def emergency_cleanup() -> None:
    """
    Perform emergency GPU memory cleanup.

    CRITICAL: Call this if system appears to be freezing.
    """
    guard = get_safety_guard()
    guard.emergency_cleanup()
