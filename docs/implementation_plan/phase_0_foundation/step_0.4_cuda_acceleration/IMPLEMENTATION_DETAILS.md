# CUDA Acceleration Implementation Details

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

---

## Table of Contents

1. [Array Model Detailed Design](#array-model-detailed-design)
2. [Base Vectorizer Detailed Design](#base-vectorizer-detailed-design)
3. [ElementWiseVectorizer Detailed Design](#elementwisevectorizer-detailed-design)
4. [TransformVectorizer Detailed Design](#transformvectorizer-detailed-design)
5. [ReductionVectorizer Detailed Design](#reductionvectorizer-detailed-design)
6. [CorrelationVectorizer Detailed Design](#correlationvectorizer-detailed-design)
7. [GridVectorizer Detailed Design](#gridvectorizer-detailed-design)
8. [Integration Points](#integration-points)
9. [Performance Considerations](#performance-considerations)

---

## Array Model Detailed Design

### Class: `CudaArray`

#### Purpose
Provide efficient block-based array storage with CPU-GPU memory management.

#### Attributes

```python
class CudaArray:
    _data: np.ndarray  # Original data (CPU)
    _gpu_data: Optional[cp.ndarray]  # GPU data (CuPy)
    _block_size: Optional[int]  # Block size for processing
    _device: str  # Current device: "cpu" or "cuda"
    _shape: Tuple[int, ...]  # Array shape
    _dtype: np.dtype  # Data type
    _num_blocks: int  # Number of blocks
    _block_shape: Tuple[int, ...]  # Shape of each block
```

#### Methods

##### `__init__(data, block_size=None, device="cpu")`

**Purpose:** Initialize array model.

**Parameters:**
- `data: np.ndarray` - Input numpy array
- `block_size: Optional[int]` - Block size (None = auto-detect based on GPU memory)
- `device: str` - Initial device ("cpu" or "cuda")

**Implementation:**
1. Store original data on CPU
2. Calculate optimal block size if not provided:
   - If GPU available: `block_size = min(available_gpu_memory // (data.nbytes * 2), data.size)`
   - If CPU only: `block_size = data.size // 4` (4 blocks)
3. Calculate number of blocks and block shape
4. Initialize GPU data as None

##### `swap_to_gpu() -> None`

**Purpose:** Move array to GPU memory.

**Implementation:**
1. Check if CuPy available
2. If `_gpu_data` is None, create CuPy array from CPU data
3. Set `_device = "cuda"`
4. Optionally free CPU memory if memory-constrained

**Error Handling:**
- If GPU unavailable, raise `CudaUnavailableError`
- If out of memory, raise `CudaMemoryError`

##### `swap_to_cpu() -> None`

**Purpose:** Move array to CPU memory.

**Implementation:**
1. If `_gpu_data` exists, copy to CPU
2. Update `_data` with GPU data
3. Set `_device = "cpu"`
4. Optionally free GPU memory

##### `get_block(block_idx: int) -> np.ndarray`

**Purpose:** Get specific block for processing.

**Parameters:**
- `block_idx: int` - Block index (0 to num_blocks-1)

**Returns:**
- `np.ndarray` - Block data (on current device)

**Implementation:**
1. Calculate block slice indices
2. Extract block from current device data
3. Return block (as numpy array if CPU, CuPy array if GPU)

##### `set_block(block_idx: int, block_data: np.ndarray) -> None`

**Purpose:** Set block data after processing.

**Parameters:**
- `block_idx: int` - Block index
- `block_data: np.ndarray` - Processed block data

**Implementation:**
1. Validate block shape matches expected
2. Calculate block slice indices
3. Copy block_data to appropriate position in array
4. Update array on current device

##### `process_blocks(operation: Callable, use_gpu: bool = True) -> CudaArray`

**Purpose:** Process array in blocks using operation.

**Parameters:**
- `operation: Callable` - Operation function (takes block, returns processed block)
- `use_gpu: bool` - Use GPU if available

**Returns:**
- `CudaArray` - New array with processed data

**Implementation:**
1. Create output array with same shape
2. If `use_gpu` and GPU available:
   - Swap input to GPU
   - Process each block on GPU
   - Swap result to CPU
3. Else:
   - Process each block on CPU
4. Return new CudaArray

##### `use_whole_array() -> np.ndarray`

**Purpose:** Get whole array as contiguous block (for FFT).

**Returns:**
- `np.ndarray` - Whole array (on current device)

**Implementation:**
1. If on GPU, return `_gpu_data` (already contiguous)
2. If on CPU, ensure `_data` is contiguous (use `np.ascontiguousarray`)
3. Return array

**Note:** This bypasses block processing - use only for operations requiring whole array.

#### Block Size Calculation

**Auto-detection algorithm:**
```python
def _calculate_block_size(self) -> int:
    """Calculate optimal block size."""
    if self._device == "cuda" and cp is not None:
        # Get available GPU memory
        mempool = cp.get_default_memory_pool()
        free_mem = mempool.free_bytes()
        
        # Reserve 50% for operations (need input + output)
        available = free_mem * 0.5
        
        # Calculate block size (bytes per element)
        bytes_per_element = self._data.itemsize
        
        # Maximum elements per block
        max_elements = available // (bytes_per_element * 2)  # *2 for input+output
        
        # Use at least 4 blocks, at most available memory
        block_size = min(max_elements, self._data.size // 4)
        return max(block_size, 1024)  # Minimum 1024 elements
    else:
        # CPU: use 4 blocks
        return self._data.size // 4
```

---

## Base Vectorizer Detailed Design

### Class: `BaseVectorizer`

#### Purpose
Abstract base class for all vectorizer implementations.

#### Attributes

```python
class BaseVectorizer(ABC):
    _use_gpu: bool  # Use GPU if available
    _block_size: Optional[int]  # Block size (None = auto)
    _whole_array: bool  # Use whole array mode
    _device: str  # Current device
    _cuda_available: bool  # CUDA availability
```

#### Methods

##### `__init__(use_gpu=True, block_size=None, whole_array=False)`

**Purpose:** Initialize vectorizer.

**Parameters:**
- `use_gpu: bool` - Use GPU acceleration
- `block_size: Optional[int]` - Block size (None = auto)
- `whole_array: bool` - Use whole array mode (no blocking)

**Implementation:**
1. Check CUDA availability: `cp is not None and cp.cuda.is_available()`
2. Set `_cuda_available`
3. If `whole_array=True`, set `_block_size=None`
4. Store configuration

##### `vectorize(array: CudaArray, *args, **kwargs) -> CudaArray`

**Purpose:** Vectorize operation on array.

**Implementation:**
1. If `_whole_array=True`:
   - Get whole array
   - Call `_process_whole_array()`
   - Create new CudaArray with result
2. Else:
   - Use `array.process_blocks()` with `_process_block()`
   - Return processed array

##### `batch(arrays: List[CudaArray], *args, **kwargs) -> List[CudaArray]`

**Purpose:** Batch process multiple arrays.

**Implementation:**
1. If `_whole_array=True`:
   - Process each array with `vectorize()`
2. Else:
   - Process arrays in parallel batches (if GPU available)
   - Use CUDA streams for parallel processing

##### `_process_block(block: np.ndarray, *args, **kwargs) -> np.ndarray`

**Purpose:** Process single block (abstract, implemented by subclasses).

**Must be implemented by subclasses.**

##### `_process_whole_array(array: np.ndarray, *args, **kwargs) -> np.ndarray`

**Purpose:** Process whole array (abstract, implemented by subclasses).

**Must be implemented by subclasses.**

**Note:** Default implementation calls `_process_block()` on whole array, but subclasses should override for efficiency.

---

## ElementWiseVectorizer Detailed Design

### Class: `ElementWiseVectorizer`

#### Purpose
Vectorize element-wise operations (arithmetic, mathematical functions).

#### Supported Operations

**Arithmetic:**
- `add(a, b)` - Addition
- `subtract(a, b)` - Subtraction
- `multiply(a, b)` - Multiplication
- `divide(a, b)` - Division
- `power(a, b)` - Power (a ** b)

**Mathematical:**
- `sin(x)`, `cos(x)`, `tan(x)`
- `exp(x)`, `log(x)`, `log10(x)`
- `sqrt(x)`, `abs(x)`
- `sign(x)`

**Comparison:**
- `less(a, b)`, `greater(a, b)`
- `less_equal(a, b)`, `greater_equal(a, b)`
- `equal(a, b)`, `not_equal(a, b)`

#### Methods

##### `vectorize_operation(array: CudaArray, operation: str, operand, *args, **kwargs) -> CudaArray`

**Purpose:** Apply element-wise operation.

**Parameters:**
- `array: CudaArray` - Input array
- `operation: str` - Operation name ("add", "multiply", "sin", etc.)
- `operand` - Second operand (for binary operations) or None (for unary)

**Implementation:**
1. Map operation name to function
2. If GPU available and `_use_gpu=True`:
   - Use CuPy operations
3. Else:
   - Use NumPy operations
4. Process in blocks or whole array based on `_whole_array`

**Example Usage:**
```python
vectorizer = ElementWiseVectorizer(use_gpu=True)
array = CudaArray(frequency_data)

# Convert frequency to multipole: l = π * D * ω
multipole = vectorizer.vectorize_operation(
    array, "multiply", np.pi * D
)
```

---

## TransformVectorizer Detailed Design

### Class: `TransformVectorizer`

#### Purpose
Vectorize transform operations requiring whole arrays (FFT, spherical harmonics).

#### Supported Operations

**FFT:**
- `fft(x)` - Forward FFT
- `ifft(x)` - Inverse FFT
- `rfft(x)` - Real FFT
- `irfft(x)` - Inverse real FFT

**Spherical Harmonics:**
- `sph_harm_synthesis(alm, lmax)` - Synthesis (alm → map)
- `sph_harm_analysis(map, lmax)` - Analysis (map → alm)

#### Methods

##### `vectorize_transform(array: CudaArray, transform: str, *args, **kwargs) -> CudaArray`

**Purpose:** Apply transform operation.

**Parameters:**
- `array: CudaArray` - Input array
- `transform: str` - Transform name ("fft", "ifft", "sph_harm_synthesis", etc.)
- `*args, **kwargs` - Transform-specific parameters

**Implementation:**
1. **Always uses whole array mode** (transforms require contiguous data)
2. Get whole array using `array.use_whole_array()`
3. If GPU available:
   - Use CuPy FFT: `cp.fft.fft()`
   - Use GPU-accelerated spherical harmonics (if available)
4. Else:
   - Use NumPy FFT: `np.fft.fft()`
   - Use CPU spherical harmonics
5. Return new CudaArray with result

**Note:** This class always sets `_whole_array=True` internally.

---

## ReductionVectorizer Detailed Design

### Class: `ReductionVectorizer`

#### Purpose
Vectorize reduction operations (sum, mean, max, etc.).

#### Supported Operations

**Standard Reductions:**
- `sum(array, axis=None)` - Sum
- `mean(array, axis=None)` - Mean
- `std(array, axis=None)` - Standard deviation
- `var(array, axis=None)` - Variance
- `max(array, axis=None)` - Maximum
- `min(array, axis=None)` - Minimum
- `argmax(array, axis=None)` - Index of maximum
- `argmin(array, axis=None)` - Index of minimum

**Logical Reductions:**
- `any(array, axis=None)` - Any true
- `all(array, axis=None)` - All true

#### Methods

##### `vectorize_reduction(array: CudaArray, reduction: str, axis=None, *args, **kwargs) -> Union[CudaArray, float]`

**Purpose:** Apply reduction operation.

**Parameters:**
- `array: CudaArray` - Input array
- `reduction: str` - Reduction name ("sum", "mean", "max", etc.)
- `axis: Optional[int]` - Axis along which to reduce (None = all axes)

**Returns:**
- `Union[CudaArray, float]` - Reduced array or scalar

**Implementation:**
1. If `axis=None` (reduce all):
   - Process in blocks, accumulate result
   - Final reduction on accumulated results
2. If `axis` specified:
   - Process along axis (may require whole array depending on axis)
3. Use GPU operations if available

---

## CorrelationVectorizer Detailed Design

### Class: `CorrelationVectorizer`

#### Purpose
Vectorize correlation and convolution operations.

#### Supported Operations

**Correlation:**
- `cross_correlation(a, b)` - Cross-correlation
- `auto_correlation(a)` - Auto-correlation
- `correlation_function(a, b, scales)` - Correlation function at scales

**Convolution:**
- `convolve(a, kernel)` - Convolution

#### Methods

##### `vectorize_correlation(array1: CudaArray, array2: CudaArray, method: str = "fft", *args, **kwargs) -> CudaArray`

**Purpose:** Calculate correlation between two arrays.

**Parameters:**
- `array1: CudaArray` - First array
- `array2: CudaArray` - Second array
- `method: str` - Method ("fft" or "direct")

**Implementation:**
1. If `method="fft"`:
   - Use FFT-based correlation (faster for large arrays)
   - Requires whole arrays
   - Use `TransformVectorizer` for FFT
2. If `method="direct"`:
   - Use direct correlation (slower but more memory-efficient)
   - Can use block processing
3. Return correlation result

**FFT-based Correlation Algorithm:**
```python
def fft_cross_correlation(a, b):
    # Pad arrays to avoid circular correlation
    size = a.size + b.size - 1
    a_padded = pad_to_size(a, size)
    b_padded = pad_to_size(b, size)
    
    # FFT
    A = fft(a_padded)
    B = fft(b_padded)
    
    # Cross-correlation in frequency domain
    C = A * conjugate(B)
    
    # Inverse FFT
    correlation = ifft(C)
    
    return correlation
```

---

## GridVectorizer Detailed Design

### Class: `GridVectorizer`

#### Purpose
Vectorize grid-based operations (local minima, gradients, etc.).

#### Supported Operations

**Detection:**
- `local_minima(grid, neighborhood_size=3)` - Local minimum detection
- `local_maxima(grid, neighborhood_size=3)` - Local maximum detection

**Gradients:**
- `gradient(grid)` - Gradient calculation
- `laplacian(grid)` - Laplacian calculation

**Curvature:**
- `curvature(grid)` - Curvature calculation

**Neighborhood:**
- `neighborhood_mean(grid, radius)` - Mean in neighborhood
- `neighborhood_std(grid, radius)` - Std in neighborhood

#### Methods

##### `vectorize_grid_operation(array: CudaArray, operation: str, *args, **kwargs) -> CudaArray`

**Purpose:** Apply grid-based operation.

**Parameters:**
- `array: CudaArray` - Input grid array
- `operation: str` - Operation name ("local_minima", "gradient", etc.)
- `*args, **kwargs` - Operation-specific parameters

**Implementation:**
1. Grid operations typically require neighborhood access
2. Use block processing with overlap (to handle boundaries)
3. For GPU:
   - Use CuPy convolution for neighborhood operations
   - Use custom CUDA kernels for complex operations
4. Return result array

**Local Minima Detection Algorithm:**
```python
def local_minima_gpu(grid, neighborhood_size=3):
    # Create neighborhood kernel
    kernel = cp.ones((neighborhood_size, neighborhood_size))
    kernel[neighborhood_size//2, neighborhood_size//2] = 0
    
    # Find minimum in neighborhood
    neighborhood_min = cp.minimum_filter(grid, size=neighborhood_size)
    
    # Points where grid equals neighborhood minimum are local minima
    is_minimum = (grid == neighborhood_min)
    
    return is_minimum
```

---

## Integration Points

### Phase 1: Θ-Field Data Processing

#### Step 1.1: Data Loader
```python
# Load data into CudaArray
data = load_theta_data()
array = CudaArray(data, device="cpu")
```

#### Step 1.2: Node Processing
```python
# Depth calculation: Δω = ω - ω_min
vectorizer = ElementWiseVectorizer(use_gpu=True)
depth = vectorizer.vectorize_operation(omega_array, "subtract", omega_min_array)

# Temperature conversion: ΔT = (Δω/ω_CMB) * T_0
temp = vectorizer.vectorize_operation(
    depth, "multiply", T_0 / omega_CMB
)
```

#### Step 1.4: Node Map Generation
```python
# Local minimum detection
grid_vectorizer = GridVectorizer(use_gpu=True)
node_mask = grid_vectorizer.vectorize_grid_operation(
    omega_field, "local_minima", neighborhood_size=5
)
```

### Phase 2: CMB Map Reconstruction

#### Step 2.1: Reconstruction Core
```python
# Frequency spectrum integration (element-wise)
vectorizer = ElementWiseVectorizer(use_gpu=True)
integrated_spectrum = vectorizer.vectorize_operation(
    frequency_spectrum, "multiply", time_factor
)

# Spherical harmonic synthesis (transform)
transform_vectorizer = TransformVectorizer(use_gpu=True)
cmb_map = transform_vectorizer.vectorize_transform(
    alm_array, "sph_harm_synthesis", lmax=l_max
)
```

### Phase 3: Power Spectrum

#### Step 3.1: Power Spectrum Calculation
```python
# Frequency to multipole: l = π * D * ω
vectorizer = ElementWiseVectorizer(use_gpu=True)
multipole = vectorizer.vectorize_operation(
    frequency_array, "multiply", np.pi * D
)

# Power spectrum: C_l ∝ ρ_Θ(ω(ℓ)) / ℓ²
power = vectorizer.vectorize_operation(
    rho_theta, "divide", multipole ** 2
)
```

### Phase 4: CMB-LSS Correlation

#### Step 4.1: Correlation Analysis
```python
# Cross-correlation using FFT
correlation_vectorizer = CorrelationVectorizer(use_gpu=True)
correlation = correlation_vectorizer.vectorize_correlation(
    cmb_map_array, lss_map_array, method="fft"
)
```

---

## Performance Considerations

### Memory Management

1. **Block Size:** Auto-detect based on available GPU memory
2. **Swap Strategy:** Only swap blocks being processed, not entire arrays
3. **Memory Pooling:** Use CuPy memory pool for efficient allocation
4. **Cleanup:** Explicitly free GPU memory after operations

### GPU Utilization

1. **Kernel Fusion:** Combine multiple operations into single kernel when possible
2. **Stream Processing:** Use CUDA streams for parallel block processing
3. **Async Operations:** Use async memory transfers when possible

### CPU Fallback

1. **Automatic Fallback:** Detect GPU unavailability and fallback to CPU
2. **Performance Warning:** Log warning when CPU fallback is used
3. **Consistent API:** Same API works on CPU and GPU

### Profiling

1. **Timing:** Measure operation time for performance analysis
2. **Memory Tracking:** Track GPU memory usage
3. **Profiling Tools:** Integration with CUDA profiling tools (nvprof, Nsight)

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com
