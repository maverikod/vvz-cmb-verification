"""
GPU memory watchdog for monitoring and managing GPU memory usage.

Tracks all processes using GPU memory and kills memory-intensive processes
when total usage exceeds 80% limit.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import threading
import time
import uuid
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime

from utils.cuda.exceptions import (
    CudaMemoryLimitExceededError,
    CudaProcessKilledError,
    CudaProcessRegistrationError,
)

# Try to import CuPy
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


@dataclass
class MemoryProcess:
    """
    Information about a process using GPU memory.

    Attributes:
        process_id: Unique process identifier
        memory_bytes: Memory used by this process in bytes
        priority: Process priority (lower = higher priority)
        created_at: When process was registered
        last_updated: Last memory update time
        description: Human-readable description
    """

    process_id: str
    memory_bytes: int
    priority: int = 5  # Default priority (1-10, lower = higher)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    description: str = ""


class MemoryWatchdog:
    """
    GPU memory watchdog for monitoring and managing memory usage.

    Tracks all registered processes and kills memory-intensive processes
    when total usage exceeds 80% limit.

    Thread-safe singleton implementation.
    """

    _instance: Optional["MemoryWatchdog"] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize memory watchdog."""
        if MemoryWatchdog._instance is not None:
            raise RuntimeError("MemoryWatchdog is a singleton. Use get_instance()")

        self._processes: Dict[str, MemoryProcess] = {}
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._memory_limit_percent = 80.0
        self._check_interval = 0.5  # Check every 0.5 seconds for aggressive monitoring

    @classmethod
    def get_instance(cls) -> "MemoryWatchdog":
        """
        Get singleton instance of MemoryWatchdog.

        Returns:
            MemoryWatchdog instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_process(
        self,
        process_id: Optional[str] = None,
        memory_bytes: int = 0,
        priority: int = 5,
        description: str = "",
    ) -> str:
        """
        Register a process with the memory watchdog.

        Args:
            process_id: Optional process ID (generated if None)
            memory_bytes: Initial memory usage in bytes
            priority: Process priority (1-10, lower = higher)
            description: Human-readable description

        Returns:
            Process ID

        Raises:
            CudaProcessRegistrationError: If registration fails
        """
        if not CUPY_AVAILABLE:
            # In non-CUDA environment, return dummy ID
            return process_id or str(uuid.uuid4())

        if process_id is None:
            process_id = str(uuid.uuid4())

        with self._lock:
            # Check if process already exists
            if process_id in self._processes:
                # Update existing process
                self._processes[process_id].memory_bytes = memory_bytes
                self._processes[process_id].last_updated = datetime.now()
                if description:
                    self._processes[process_id].description = description
                return process_id

            # Check memory limit before registration
            try:
                self._check_memory_limit(memory_bytes)
            except CudaMemoryLimitExceededError as e:
                raise CudaProcessRegistrationError(
                    process_id, f"Memory limit check failed: {e.message}"
                ) from e

            # Register new process
            process = MemoryProcess(
                process_id=process_id,
                memory_bytes=memory_bytes,
                priority=priority,
                description=description,
            )
            self._processes[process_id] = process

            # Start monitoring if not already started
            if not self._monitoring:
                self._start_monitoring()

        return process_id

    def unregister_process(self, process_id: str) -> None:
        """
        Unregister a process from the memory watchdog.

        Args:
            process_id: Process ID to unregister
        """
        with self._lock:
            if process_id in self._processes:
                del self._processes[process_id]

    def update_process_memory(self, process_id: str, memory_bytes: int) -> None:
        """
        Update memory usage for a registered process.

        CRITICAL: Checks memory limit and kills process if limit would be exceeded.

        Args:
            process_id: Process ID
            memory_bytes: New memory usage in bytes

        Raises:
            CudaProcessKilledError: If process is killed due to memory limit
        """
        if not CUPY_AVAILABLE:
            return

        with self._lock:
            if process_id not in self._processes:
                return

            # CRITICAL: Check memory limit BEFORE update to prevent system freeze
            try:
                # Calculate additional memory needed
                old_memory = self._processes[process_id].memory_bytes
                additional_bytes = memory_bytes - old_memory
                self._check_memory_limit(additional_bytes, exclude_process=process_id)
            except CudaMemoryLimitExceededError:
                # Kill this process immediately if it would exceed limit
                self._kill_process(process_id)
                # Force cleanup
                self._force_cleanup()
                raise CudaProcessKilledError(
                    process_id, "Memory usage would exceed 80% limit - process killed"
                )

            # Update memory usage only if check passed
            self._processes[process_id].memory_bytes = memory_bytes
            self._processes[process_id].last_updated = datetime.now()

    def _check_memory_limit(
        self, additional_bytes: int = 0, exclude_process: Optional[str] = None
    ) -> None:
        """
        Check if memory usage would exceed 80% limit.

        CRITICAL: This method actively kills processes if limit would be exceeded
        to prevent system hang/freeze.

        Args:
            additional_bytes: Additional memory to check
            exclude_process: Process ID to exclude from calculation

        Raises:
            CudaMemoryLimitExceededError: If limit would be exceeded
        """
        if not CUPY_AVAILABLE:
            return

        try:
            # Get total GPU memory
            # getDeviceProperties returns dict in newer CuPy versions
            device_props = cp.cuda.runtime.getDeviceProperties(0)
            total_mem = (
                device_props["totalGlobalMem"]
                if isinstance(device_props, dict)
                else device_props.totalGlobalMem
            )
            max_allowed = total_mem * (self._memory_limit_percent / 100.0)

            # Get actual GPU memory usage (more accurate than registered)
            mempool = cp.get_default_memory_pool()
            actual_used = mempool.used_bytes()

            # Calculate current usage from registered processes
            registered_usage = sum(
                p.memory_bytes
                for pid, p in self._processes.items()
                if pid != exclude_process
            )

            # Use maximum of actual and registered to be safe
            current_usage = max(actual_used, registered_usage)

            # Add additional memory
            total_usage = current_usage + additional_bytes

            # Check limit - CRITICAL: kill processes BEFORE limit is reached
            if total_usage > max_allowed * 0.9:  # Start killing at 90% of 80% = 72%
                # Aggressively kill memory-intensive processes
                killed = self._kill_memory_intensive_processes()
                if killed:
                    # Recalculate after killing
                    current_usage = sum(
                        p.memory_bytes
                        for pid, p in self._processes.items()
                        if pid != exclude_process
                    )
                    total_usage = current_usage + additional_bytes

            # Final check - if still over limit, raise error
            if total_usage > max_allowed:
                usage_percent = (total_usage / total_mem) * 100
                # Force cleanup before raising error
                self._force_cleanup()
                raise CudaMemoryLimitExceededError(
                    current_usage=usage_percent,
                    required_bytes=additional_bytes,
                    total_memory=total_mem,
                )
        except CudaMemoryLimitExceededError:
            raise
        except Exception:
            # If check fails, be conservative and raise error
            raise CudaMemoryLimitExceededError(
                current_usage=100.0,
                required_bytes=additional_bytes,
                total_memory=0,
                message="Memory check failed - refusing allocation for safety",
            )

    def _kill_process(self, process_id: str) -> None:
        """
        Kill a memory-intensive process.

        Args:
            process_id: Process ID to kill
        """
        with self._lock:
            if process_id in self._processes:
                # Remove from registry
                del self._processes[process_id]

    def _kill_memory_intensive_processes(self) -> List[str]:
        """
        Kill memory-intensive processes when limit is exceeded.

        CRITICAL: Aggressively kills processes to prevent system freeze.

        Returns:
            List of killed process IDs
        """
        if not CUPY_AVAILABLE:
            return []

        killed = []
        try:
            total_mem = cp.cuda.runtime.getDeviceProperties(0).totalGlobalMem
            max_allowed = total_mem * (self._memory_limit_percent / 100.0)

            # Get actual GPU memory usage
            mempool = cp.get_default_memory_pool()
            actual_used = mempool.used_bytes()

            # Calculate current usage (use max of actual and registered)
            registered_usage = sum(p.memory_bytes for p in self._processes.values())
            current_usage = max(actual_used, registered_usage)

            # Kill if over 70% of limit (56% of total) - be aggressive
            kill_threshold = max_allowed * 0.7

            if current_usage > kill_threshold:
                # Sort processes by priority (lower = higher priority) and memory
                # Kill low-priority, high-memory processes first
                sorted_processes = sorted(
                    self._processes.items(),
                    key=lambda x: (x[1].priority, -x[1].memory_bytes),
                )

                # Kill processes until well under limit (target 50% of limit)
                target_usage = max_allowed * 0.5

                for process_id, process in sorted_processes:
                    if current_usage <= target_usage:
                        break

                    # Kill this process
                    self._kill_process(process_id)
                    killed.append(process_id)
                    # Estimate reduction (use registered memory)
                    current_usage -= process.memory_bytes

                # Force GPU memory cleanup after killing
                if killed:
                    try:
                        mempool.free_all_blocks()
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass

        except Exception:
            # If monitoring fails, try to free all memory as last resort
            try:
                if CUPY_AVAILABLE:
                    cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

        return killed

    def _force_cleanup(self) -> None:
        """
        Force cleanup of GPU memory.

        CRITICAL: Called when memory limit is exceeded to prevent system freeze.
        """
        if not CUPY_AVAILABLE:
            return

        try:
            # Kill all low-priority processes
            low_priority = [
                (pid, p)
                for pid, p in self._processes.items()
                if p.priority >= 5  # Kill priority 5 and above
            ]
            for process_id, _ in low_priority:
                self._kill_process(process_id)

            # Force GPU memory pool cleanup
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            cp.get_default_memory_pool().free_all_blocks()

            # Clear all registered processes
            self._processes.clear()
        except Exception:
            # Ignore errors during emergency cleanup
            pass

    def _start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """
        Background monitoring loop.

        CRITICAL: Aggressively monitors and kills processes to prevent system freeze.
        """
        while self._monitoring:
            try:
                # Check and kill memory-intensive processes
                killed = self._kill_memory_intensive_processes()
                if killed:
                    # Force cleanup after killing
                    self._force_cleanup()

                # Shorter interval for more aggressive monitoring
                time.sleep(self._check_interval)
            except Exception:
                # On any error, try emergency cleanup
                try:
                    self._force_cleanup()
                except Exception:
                    pass
                time.sleep(self._check_interval)

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        if not CUPY_AVAILABLE:
            return {
                "total_memory": 0,
                "used_memory": 0,
                "free_memory": 0,
                "usage_percent": 0.0,
                "process_count": 0,
            }

        try:
            total_mem = cp.cuda.runtime.getDeviceProperties(0).totalGlobalMem
            mempool = cp.get_default_memory_pool()
            used_mem = mempool.used_bytes()
            registered_usage = sum(p.memory_bytes for p in self._processes.values())

            return {
                "total_memory": total_mem,
                "used_memory": used_mem,
                "registered_usage": registered_usage,
                "free_memory": total_mem - used_mem,
                "usage_percent": (used_mem / total_mem) * 100.0,
                "process_count": len(self._processes),
            }
        except Exception:
            return {
                "total_memory": 0,
                "used_memory": 0,
                "registered_usage": 0,
                "free_memory": 0,
                "usage_percent": 0.0,
                "process_count": len(self._processes),
            }

    def get_registered_processes(self) -> List[Dict[str, Any]]:
        """
        Get list of all registered processes.

        Returns:
            List of process information dictionaries
        """
        with self._lock:
            return [
                {
                    "process_id": p.process_id,
                    "memory_bytes": p.memory_bytes,
                    "priority": p.priority,
                    "created_at": p.created_at.isoformat(),
                    "last_updated": p.last_updated.isoformat(),
                    "description": p.description,
                }
                for p in self._processes.values()
            ]
