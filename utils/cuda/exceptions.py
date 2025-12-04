"""
CUDA-related exceptions.

All CUDA exceptions are defined here for centralized error handling.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""


class CudaError(Exception):
    """Base exception for all CUDA-related errors."""

    pass


class CudaUnavailableError(CudaError):
    """Raised when CUDA is requested but unavailable."""

    def __init__(self, message: str = "CUDA is not available"):
        """
        Initialize exception.

        Args:
            message: Error message
        """
        super().__init__(message)
        self.message = message


class CudaMemoryError(CudaError):
    """Raised when CUDA memory operations fail."""

    def __init__(self, message: str = "CUDA memory operation failed"):
        """
        Initialize exception.

        Args:
            message: Error message
        """
        super().__init__(message)
        self.message = message


class CudaMemoryLimitExceededError(CudaError):
    """Raised when GPU memory usage exceeds 80% limit."""

    def __init__(
        self,
        current_usage: float,
        required_bytes: int,
        total_memory: int,
        message: str = "",
    ):
        """
        Initialize exception.

        Args:
            current_usage: Current memory usage percentage
            required_bytes: Required memory in bytes
            total_memory: Total GPU memory in bytes
            message: Optional custom error message
        """
        if not message:
            message = (
                f"GPU memory usage would exceed 80% limit. "
                f"Current: {current_usage:.2f}%, "
                f"Required: {required_bytes / (1024**3):.2f} GB, "
                f"Total GPU memory: {total_memory / (1024**3):.2f} GB"
            )
        super().__init__(message)
        self.current_usage = current_usage
        self.required_bytes = required_bytes
        self.total_memory = total_memory
        self.message = message


class CudaProcessKilledError(CudaError):
    """Raised when a CUDA process is killed by memory watchdog."""

    def __init__(self, process_id: str, reason: str = ""):
        """
        Initialize exception.

        Args:
            process_id: ID of the killed process
            reason: Reason for killing the process
        """
        message = f"CUDA process {process_id} was killed by memory watchdog"
        if reason:
            message += f": {reason}"
        super().__init__(message)
        self.process_id = process_id
        self.reason = reason
        self.message = message


class CudaProcessRegistrationError(CudaError):
    """Raised when process registration with memory watchdog fails."""

    def __init__(self, process_id: str, message: str = ""):
        """
        Initialize exception.

        Args:
            process_id: ID of the process that failed to register
            message: Error message
        """
        if not message:
            message = f"Failed to register process {process_id} with memory watchdog"
        super().__init__(message)
        self.process_id = process_id
        self.message = message


class CudaVectorizationError(CudaError):
    """Raised when vectorization operation fails."""

    def __init__(self, operation: str, message: str = ""):
        """
        Initialize exception.

        Args:
            operation: Name of the failed operation
            message: Error message
        """
        if not message:
            message = f"Vectorization operation '{operation}' failed"
        super().__init__(message)
        self.operation = operation
        self.message = message


class CudaBlockProcessingError(CudaError):
    """Raised when block processing fails."""

    def __init__(self, block_index: int, message: str = ""):
        """
        Initialize exception.

        Args:
            block_index: Index of the failed block
            message: Error message
        """
        if not message:
            message = f"Block processing failed for block {block_index}"
        super().__init__(message)
        self.block_index = block_index
        self.message = message
