"""
Element-wise vectorizer for CUDA acceleration.

Provides vectorized element-wise operations (arithmetic, mathematical functions).

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Union, Any, TYPE_CHECKING, List

if TYPE_CHECKING:
    from utils.cuda.array_model import CudaArray

import numpy as np

from utils.cuda.vectorizer import BaseVectorizer

# Try to import CuPy
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class ElementWiseVectorizer(BaseVectorizer):
    """
    Vectorizer for element-wise operations.

    Supports:
    - Arithmetic: +, -, *, /, **
    - Mathematical: sin, cos, exp, log, sqrt, abs
    - Comparison: <, >, <=, >=, ==, !=

    Attributes:
        _operation_map: Mapping of operation names to functions
    """

    def __init__(
        self,
        use_gpu: bool = True,
        block_size: Optional[int] = None,
        whole_array: bool = False,
    ):
        """
        Initialize element-wise vectorizer.

        Args:
            use_gpu: Use GPU acceleration if available
            block_size: Block size (None = auto)
            whole_array: Use whole array mode (default: False for element-wise)
        """
        super().__init__(
            use_gpu=use_gpu, block_size=block_size, whole_array=whole_array
        )
        self._operation_map = self._build_operation_map()

    def _build_operation_map(self) -> dict:
        """
        Build mapping of operation names to functions.

        Returns:
            Dictionary mapping operation names to functions
        """
        if self.use_gpu and CUPY_AVAILABLE:
            # Use CuPy operations
            return {
                "add": cp.add,
                "subtract": cp.subtract,
                "multiply": cp.multiply,
                "divide": cp.divide,
                "power": cp.power,
                "sin": cp.sin,
                "cos": cp.cos,
                "tan": cp.tan,
                "exp": cp.exp,
                "log": cp.log,
                "log10": cp.log10,
                "sqrt": cp.sqrt,
                "abs": cp.abs,
                "sign": cp.sign,
                "less": cp.less,
                "greater": cp.greater,
                "less_equal": cp.less_equal,
                "greater_equal": cp.greater_equal,
                "equal": cp.equal,
                "not_equal": cp.not_equal,
            }
        else:
            # Use NumPy operations
            return {
                "add": np.add,
                "subtract": np.subtract,
                "multiply": np.multiply,
                "divide": np.divide,
                "power": np.power,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "exp": np.exp,
                "log": np.log,
                "log10": np.log10,
                "sqrt": np.sqrt,
                "abs": np.abs,
                "sign": np.sign,
                "less": np.less,
                "greater": np.greater,
                "less_equal": np.less_equal,
                "greater_equal": np.greater_equal,
                "equal": np.equal,
                "not_equal": np.not_equal,
            }

    def vectorize_operation(
        self,
        array: "CudaArray",  # noqa: F821
        operation: str,
        operand: Optional[Union[float, np.ndarray, "CudaArray"]] = None,  # noqa: F821
        *args: Any,
        **kwargs: Any,
    ) -> "CudaArray":  # noqa: F821
        """
        Apply element-wise operation.

        Args:
            array: Input CudaArray
            operation: Operation name ("add", "multiply", "sin", etc.)
            operand: Second operand (for binary operations) or None (for unary)
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Result CudaArray

        Raises:
            ValueError: If operation not supported
        """
        if operation not in self._operation_map:
            raise ValueError(f"Unsupported operation: {operation}")

        op_func = self._operation_map[operation]

        # Store operand for use in _process_block
        self._current_operation = operation
        self._current_operand = operand
        self._current_op_func = op_func

        return self.vectorize(array, *args, **kwargs)

    def _process_block(
        self, block: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process single block with element-wise operation.

        Args:
            block: Block data
            *args: Additional arguments (not used)
            **kwargs: Additional keyword arguments (not used)

        Returns:
            Processed block
        """
        # Get operation and operand from stored attributes
        operand = getattr(self, "_current_operand", None)
        op_func = getattr(self, "_current_op_func", None)

        if op_func is None:
            raise ValueError("Operation not set. Use vectorize_operation() method.")

        # Convert operand if needed
        if operand is not None:
            from utils.cuda.array_model import CudaArray

            if isinstance(operand, CudaArray):
                # If operand is CudaArray, get data from same device as block
                if self.use_gpu and CUPY_AVAILABLE and isinstance(block, cp.ndarray):
                    # Block is on GPU, get operand from GPU if available
                    if operand.device == "cuda":
                        operand = operand.use_whole_array()
                    else:
                        # Operand is on CPU, convert to CuPy
                        operand = cp.asarray(operand.to_numpy())
                else:
                    # Block is on CPU, get operand from CPU
                    operand = operand.to_numpy()
            elif self.use_gpu and CUPY_AVAILABLE and isinstance(block, cp.ndarray):
                # Block is CuPy array, convert operand to CuPy
                if isinstance(operand, np.ndarray):
                    operand = cp.asarray(operand)
                elif isinstance(operand, (int, float)):
                    operand = operand  # Scalar is fine
            elif isinstance(operand, np.ndarray):
                # Block is numpy, operand is numpy - both on CPU, fine
                pass

        # Apply operation
        if operand is None:
            # Unary operation
            result = op_func(block)
        else:
            # Binary operation
            result = op_func(block, operand)

        return result

    def _process_whole_array(
        self, array: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process whole array with element-wise operation.

        Args:
            array: Whole array data
            *args: Additional arguments (not used)
            **kwargs: Additional keyword arguments (not used)

        Returns:
            Processed array
        """
        # For element-wise operations, whole array is same as block processing
        return self._process_block(array, *args, **kwargs)

    def add(
        self,
        array: "CudaArray",
        operand: Union[float, np.ndarray, "CudaArray"],  # noqa: F821
    ) -> "CudaArray":  # noqa: F821
        """Add operand to array."""
        return self.vectorize_operation(array, "add", operand)

    def subtract(
        self,
        array: "CudaArray",
        operand: Union[float, np.ndarray, "CudaArray"],  # noqa: F821
    ) -> "CudaArray":  # noqa: F821
        """Subtract operand from array."""
        return self.vectorize_operation(array, "subtract", operand)

    def multiply(
        self,
        array: "CudaArray",
        operand: Union[float, np.ndarray, "CudaArray"],  # noqa: F821
    ) -> "CudaArray":  # noqa: F821
        """Multiply array by operand."""
        return self.vectorize_operation(array, "multiply", operand)

    def divide(
        self,
        array: "CudaArray",
        operand: Union[float, np.ndarray, "CudaArray"],  # noqa: F821
    ) -> "CudaArray":  # noqa: F821
        """Divide array by operand."""
        return self.vectorize_operation(array, "divide", operand)

    def power(
        self,
        array: "CudaArray",
        exponent: Union[float, np.ndarray, "CudaArray"],  # noqa: F821
    ) -> "CudaArray":  # noqa: F821
        """Raise array to power."""
        return self.vectorize_operation(array, "power", exponent)

    def sin(self, array: "CudaArray") -> "CudaArray":  # noqa: F821
        """Apply sine function."""
        return self.vectorize_operation(array, "sin", None)

    def cos(self, array: "CudaArray") -> "CudaArray":  # noqa: F821
        """Apply cosine function."""
        return self.vectorize_operation(array, "cos", None)

    def exp(self, array: "CudaArray") -> "CudaArray":  # noqa: F821
        """Apply exponential function."""
        return self.vectorize_operation(array, "exp", None)

    def log(self, array: "CudaArray") -> "CudaArray":  # noqa: F821
        """Apply natural logarithm."""
        return self.vectorize_operation(array, "log", None)

    def sqrt(self, array: "CudaArray") -> "CudaArray":  # noqa: F821
        """Apply square root."""
        return self.vectorize_operation(array, "sqrt", None)

    def abs(self, array: "CudaArray") -> "CudaArray":  # noqa: F821
        """Apply absolute value."""
        return self.vectorize_operation(array, "abs", None)

    def batch(
        self, arrays: List["CudaArray"], *args: Any, **kwargs: Any  # noqa: F821
    ) -> List["CudaArray"]:  # noqa: F821
        """
        Batch process multiple arrays.

        Args:
            arrays: List of input CudaArrays
            *args: Operation-specific arguments (operation, operand)
            **kwargs: Operation-specific keyword arguments

        Returns:
            List of result CudaArrays
        """
        # Extract operation and operand from args if provided
        if len(args) >= 2:
            operation = args[0]
            operand = args[1] if len(args) > 1 else None
            results = []
            for array in arrays:
                result = self.vectorize_operation(array, operation, operand)
                results.append(result)
            return results
        else:
            # Fallback to base implementation
            return super().batch(arrays, *args, **kwargs)
