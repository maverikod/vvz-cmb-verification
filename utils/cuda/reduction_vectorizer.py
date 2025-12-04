"""
Reduction vectorizer for CUDA acceleration.

Provides vectorized reduction operations (sum, mean, max, etc.).

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Optional, Any, Union, TYPE_CHECKING

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


class ReductionVectorizer(BaseVectorizer):
    """
    Vectorizer for reduction operations.

    Supports:
    - Standard reductions: sum, mean, std, var, max, min, argmax, argmin
    - Logical reductions: any, all
    """

    def __init__(
        self,
        use_gpu: bool = True,
        block_size: Optional[int] = None,
        whole_array: bool = False,
    ):
        """
        Initialize reduction vectorizer.

        Args:
            use_gpu: Use GPU acceleration if available
            block_size: Block size (None = auto)
            whole_array: Use whole array mode (default: False)
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
            return {
                "sum": cp.sum,
                "mean": cp.mean,
                "std": cp.std,
                "var": cp.var,
                "max": cp.max,
                "min": cp.min,
                "argmax": cp.argmax,
                "argmin": cp.argmin,
                "any": cp.any,
                "all": cp.all,
            }
        else:
            return {
                "sum": np.sum,
                "mean": np.mean,
                "std": np.std,
                "var": np.var,
                "max": np.max,
                "min": np.min,
                "argmax": np.argmax,
                "argmin": np.argmin,
                "any": np.any,
                "all": np.all,
            }

    def vectorize_reduction(
        self,
        array: "CudaArray",  # noqa: F821
        reduction: str,
        axis: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Union["CudaArray", float, int]:  # noqa: F821
        """
        Apply reduction operation.

        Args:
            array: Input CudaArray
            reduction: Reduction name ("sum", "mean", "max", etc.)
            axis: Axis along which to reduce (None = all axes)
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Reduced array or scalar

        Raises:
            ValueError: If reduction not supported
        """
        if reduction not in self._operation_map:
            raise ValueError(f"Unsupported reduction: {reduction}")

        # Store reduction type for use in processing
        self._current_reduction = reduction
        self._current_axis = axis
        self._reduction_args = args
        self._reduction_kwargs = kwargs

        # If axis=None (reduce all), process in blocks and accumulate
        if axis is None:
            return self._reduce_all_axes(array, *args, **kwargs)
        else:
            # Process along specific axis (may require whole array)
            return self._reduce_along_axis(array, axis, *args, **kwargs)

    def _reduce_all_axes(
        self, array: "CudaArray", *args: Any, **kwargs: Any  # noqa: F821
    ) -> Union[float, int]:
        """
        Reduce all axes (scalar result).

        Args:
            array: Input CudaArray
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Scalar result
        """
        reduction = getattr(self, "_current_reduction", None)
        if reduction is None:
            raise ValueError("Reduction not set")

        op_func = self._operation_map[reduction]

        # Process in blocks and accumulate
        if self._whole_array:
            whole_data = array.use_whole_array()
            result = op_func(whole_data, *args, **kwargs)
        else:
            # Process blocks and accumulate
            results = []
            for block_idx in range(array.num_blocks):
                block = array.get_block(block_idx)
                block_result = op_func(block, *args, **kwargs)
                results.append(block_result)

            # Final reduction on accumulated results
            if reduction in ["sum", "mean"]:
                result = sum(results)
                if reduction == "mean":
                    result = result / array.num_blocks
            elif reduction in ["max", "argmax"]:
                result = max(results)
            elif reduction in ["min", "argmin"]:
                result = min(results)
            else:
                # For other reductions, process whole array
                whole_data = array.use_whole_array()
                result = op_func(whole_data, *args, **kwargs)

        # Convert to Python scalar
        if self.use_gpu and CUPY_AVAILABLE and isinstance(result, cp.ndarray):
            result = float(cp.asnumpy(result))
        elif isinstance(result, np.ndarray):
            result = float(result)
        elif isinstance(result, (np.integer, np.floating)):
            result = result.item()

        return result

    def _reduce_along_axis(
        self, array: "CudaArray", axis: int, *args: Any, **kwargs: Any  # noqa: F821
    ) -> "CudaArray":  # noqa: F821
        """
        Reduce along specific axis.

        Args:
            array: Input CudaArray
            axis: Axis along which to reduce
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Reduced CudaArray
        """
        reduction = getattr(self, "_current_reduction", None)
        if reduction is None:
            raise ValueError("Reduction not set")

        op_func = self._operation_map[reduction]

        # For axis reduction, use whole array
        whole_data = array.use_whole_array()
        result = op_func(whole_data, axis=axis, *args, **kwargs)

        # Convert to numpy if needed
        if self.use_gpu and CUPY_AVAILABLE and isinstance(result, cp.ndarray):
            result_numpy = cp.asnumpy(result)
        else:
            result_numpy = np.asarray(result)

        # Create new CudaArray
        from utils.cuda.array_model import CudaArray

        return CudaArray(result_numpy, block_size=None, device=array.device)

    def _process_block(
        self, block: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process single block (not typically used for reductions).

        Args:
            block: Block data
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed block
        """
        reduction = getattr(self, "_current_reduction", None)
        axis = getattr(self, "_current_axis", None)

        if reduction is None:
            raise ValueError("Reduction not set")

        op_func = self._operation_map[reduction]

        if axis is None:
            return op_func(block, *args, **kwargs)
        else:
            return op_func(block, axis=axis, *args, **kwargs)

    def _process_whole_array(
        self, array: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Process whole array with reduction.

        Args:
            array: Whole array data
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Reduced array or scalar
        """
        reduction = getattr(self, "_current_reduction", None)
        axis = getattr(self, "_current_axis", None)

        if reduction is None:
            raise ValueError("Reduction not set")

        op_func = self._operation_map[reduction]

        if axis is None:
            return op_func(array, *args, **kwargs)
        else:
            return op_func(array, axis=axis, *args, **kwargs)
