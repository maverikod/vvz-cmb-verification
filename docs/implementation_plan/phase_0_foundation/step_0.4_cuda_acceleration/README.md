# Step 0.4: CUDA Acceleration Infrastructure

**Phase:** 0 - Foundation and Infrastructure  
**Step:** 0.4 - CUDA Acceleration Infrastructure  
**Module:** `utils/cuda/`

---

## Overview

This step implements the CUDA acceleration infrastructure for the CMB verification project. It provides:

1. **Block-based array model** with swap capabilities and block-wise processing
2. **Vectorizer and batcher base classes** for CUDA-accelerated operations
3. **Whole-array processing option** for operations like FFT that require contiguous data
4. **Specialized vectorizer subclasses** for different operation types

**Priority order (from cuda.mdc rules):**
1. Block-based processing (блочная обработка)
2. Vectorization (векторизация)
3. Batching (батчинг)
4. CUDA acceleration (на основе этого высокоприоритетное использование CUDA)

---

## Analysis of Implementation Plan for CUDA Opportunities

### Phase 1: Θ-Field Data Processing

#### Step 1.1: Θ-Field Data Loader
- **CUDA Opportunity:** Loading and initial processing of large frequency spectrum arrays
- **Operations:**
  - Array loading and memory management
  - Initial data validation and transformation
- **Vectorization:** Bulk array operations on frequency data

#### Step 1.2: Θ-Node Data Processing
- **CUDA Opportunity:** Node depth calculations on large grids
- **Operations:**
  - Depth calculation: Δω = ω - ω_min for all nodes
  - Temperature mapping: ΔT = (Δω/ω_CMB) T_0
- **Vectorization:** Element-wise operations on node arrays
- **Block Processing:** Process nodes in spatial blocks

#### Step 1.3: Θ-Field Evolution Data
- **CUDA Opportunity:** Temporal evolution processing
- **Operations:**
  - Interpolation over time arrays
  - Evolution rate calculations (numerical derivatives)
- **Vectorization:** Time-series operations

#### Step 1.4: Θ-Node Map Generation
- **CUDA Opportunity:** Local minimum detection on large grids
- **Operations:**
  - Local minimum detection: x_node = {x : ω(x) = ω_min(x)}
  - Node classification (depth, area, curvature)
  - Grid-based operations
- **Block Processing:** Process grid in spatial blocks
- **Vectorization:** Grid-wide operations

### Phase 2: CMB Map Reconstruction

#### Step 2.1: CMB Reconstruction Core
- **CUDA Opportunity:** High-priority - Large-scale map reconstruction
- **Operations:**
  - Frequency spectrum integration: ρ_Θ(ω,t) for each direction n̂
  - Depth calculation for each direction: Δω(n̂) = ω(n̂) - ω_min(n̂)
  - Temperature conversion: ΔT = (Δω/ω_CMB) T_0 for all directions
  - Spherical harmonic synthesis
- **Block Processing:** Process directions in batches
- **Vectorization:** Direction-by-direction operations
- **Whole Array:** Spherical harmonic transforms (FFT-like operations)

#### Step 2.2: CMB Map Validation
- **CUDA Opportunity:** Map comparison operations
- **Operations:**
  - Map-to-map comparison
  - Statistical calculations on large maps
- **Vectorization:** Element-wise map operations

#### Step 2.3: Node-to-Map Mapping
- **CUDA Opportunity:** Coordinate transformations
- **Operations:**
  - Projection calculations (z≈1100 to z=0)
  - Coordinate transformations for all nodes
- **Vectorization:** Batch coordinate transformations

### Phase 3: Power Spectrum Generation

#### Step 3.1: Power Spectrum Calculation
- **CUDA Opportunity:** High-priority - Large-scale spectrum calculations
- **Operations:**
  - Frequency to multipole conversion: l = π D ω (for large arrays)
  - Direct C_l calculation: C_l ∝ ρ_Θ(ω(ℓ)) / ℓ²
  - Temporal evolution integration
  - High-l range processing (up to l≈10000)
- **Block Processing:** Process multipole ranges in blocks
- **Vectorization:** Array-wide operations
- **Whole Array:** FFT operations if needed for spectrum analysis

#### Step 3.2: High-l Sub-peaks Analysis
- **CUDA Opportunity:** Peak detection and analysis
- **Operations:**
  - Peak finding algorithms
  - Sub-peak structure analysis
- **Vectorization:** Array-wide peak detection

#### Step 3.3: Spectrum Comparison
- **CUDA Opportunity:** Spectrum comparison operations
- **Operations:**
  - Spectrum-to-spectrum comparison
  - Statistical analysis
- **Vectorization:** Element-wise comparison operations

### Phase 4: CMB-LSS Correlation

#### Step 4.1: Correlation Analysis Core
- **CUDA Opportunity:** High-priority - Large-scale correlation calculations
- **Operations:**
  - Cross-correlation calculations between CMB and LSS
  - Correlation function computation at 10-12 Mpc scales
  - Angular separation calculations
- **Block Processing:** Process correlation scales in blocks
- **Vectorization:** Batch correlation calculations
- **Whole Array:** FFT-based correlation (if using FFT method)

#### Step 4.2: Phi-Split Analysis
- **CUDA Opportunity:** Signal enhancement calculations
- **Operations:**
  - φ-split technique implementation
  - Signal processing operations
- **Vectorization:** Array-wide signal processing

#### Step 4.3: Node-LSS Mapping
- **CUDA Opportunity:** Large-scale mapping operations
- **Operations:**
  - Node-to-LSS position mapping
  - Galaxy type prediction calculations
  - Correlation map generation
- **Block Processing:** Process nodes in spatial blocks
- **Vectorization:** Batch mapping operations

### Phase 5: ACT/SPT Predictions

#### Step 5.1: High-l Peak Prediction
- **CUDA Opportunity:** Peak prediction calculations
- **Operations:**
  - Peak position calculations
  - Amplitude estimation
- **Vectorization:** Array-wide prediction operations

#### Step 5.2: Frequency Invariance Test
- **CUDA Opportunity:** Multi-frequency cross-spectra
- **Operations:**
  - Cross-spectra calculations at 90-350 GHz
  - Frequency comparison operations
- **Block Processing:** Process frequency bands in blocks
- **Vectorization:** Multi-frequency operations

#### Step 5.3: Predictions Report
- **CUDA Opportunity:** Report generation (minimal)
- **Operations:**
  - Statistical aggregations
- **Vectorization:** Summary calculations

### Phase 6: Chain Verification

#### Step 6.1: Cluster Plateau Analysis
- **CUDA Opportunity:** Cluster data analysis
- **Operations:**
  - Plateau slope calculations
  - Correlation with CMB node directions
- **Vectorization:** Batch cluster operations

#### Step 6.2: Galaxy Distribution Analysis
- **CUDA Opportunity:** Galaxy distribution calculations
- **Operations:**
  - U1/U2/U3 distribution calculations
  - SWI, χ₆ calculations by node directions
- **Block Processing:** Process galaxy catalogs in blocks
- **Vectorization:** Batch galaxy operations

#### Step 6.3: Chain Verification Report
- **CUDA Opportunity:** Report generation (minimal)
- **Operations:**
  - Statistical aggregations
- **Vectorization:** Summary calculations

---

## Architecture Design

### 1. Array Model (`utils/cuda/array_model.py`)

**Purpose:** Provide block-based array storage with swap capabilities and efficient memory management.

#### Core Features:
- **Block-based storage:** Arrays divided into manageable blocks
- **Swap capability:** Move data between CPU and GPU memory
- **Block-wise processing:** Process arrays in blocks for memory efficiency
- **Whole-array option:** Option to use array as contiguous block (for FFT)

#### Class Structure:

```python
class CudaArray:
    """
    Block-based array model for CUDA acceleration.
    
    Supports:
    - Block-based processing
    - CPU-GPU memory swap
    - Whole-array mode (for FFT operations)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        block_size: Optional[int] = None,
        device: str = "cpu"
    ):
        """
        Initialize array model.
        
        Args:
            data: Input numpy array
            block_size: Size of blocks for processing (None = auto)
            device: "cpu" or "cuda"
        """
        pass
    
    def swap_to_gpu(self) -> None:
        """Move array to GPU memory."""
        pass
    
    def swap_to_cpu(self) -> None:
        """Move array to CPU memory."""
        pass
    
    def get_block(self, block_idx: int) -> np.ndarray:
        """Get specific block for processing."""
        pass
    
    def set_block(self, block_idx: int, block_data: np.ndarray) -> None:
        """Set block data after processing."""
        pass
    
    def process_blocks(
        self,
        operation: Callable,
        use_gpu: bool = True
    ) -> "CudaArray":
        """Process array in blocks using operation."""
        pass
    
    def use_whole_array(self) -> np.ndarray:
        """Get whole array as contiguous block (for FFT)."""
        pass
```

### 2. Vectorizer Base Class (`utils/cuda/vectorizer.py`)

**Purpose:** Base class for vectorizing operations on arrays. Subclasses implement specific operation types.

#### Core Features:
- **Vectorization:** Apply operations to entire arrays or blocks
- **Batching:** Process multiple arrays together
- **GPU acceleration:** Automatic GPU usage when available
- **Block processing:** Support for block-wise operations
- **Whole-array mode:** Support for operations requiring contiguous arrays

#### Class Structure:

```python
class BaseVectorizer(ABC):
    """
    Base class for vectorizing operations on arrays.
    
    Subclasses implement specific operation types:
    - ElementWiseVectorizer: Element-wise operations (+, -, *, /, etc.)
    - TransformVectorizer: Transform operations (FFT, spherical harmonics)
    - ReductionVectorizer: Reduction operations (sum, mean, max, etc.)
    - CorrelationVectorizer: Correlation operations
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        block_size: Optional[int] = None,
        whole_array: bool = False
    ):
        """
        Initialize vectorizer.
        
        Args:
            use_gpu: Use GPU acceleration if available
            block_size: Block size for processing (None = auto)
            whole_array: Use whole array (no blocking) for operations
        """
        pass
    
    @abstractmethod
    def vectorize(
        self,
        array: CudaArray,
        *args,
        **kwargs
    ) -> CudaArray:
        """
        Vectorize operation on array.
        
        Args:
            array: Input CudaArray
            *args: Operation-specific arguments
            **kwargs: Operation-specific keyword arguments
        
        Returns:
            Result CudaArray
        """
        raise NotImplementedError
    
    @abstractmethod
    def batch(
        self,
        arrays: List[CudaArray],
        *args,
        **kwargs
    ) -> List[CudaArray]:
        """
        Batch process multiple arrays.
        
        Args:
            arrays: List of input CudaArrays
            *args: Operation-specific arguments
            **kwargs: Operation-specific keyword arguments
        
        Returns:
            List of result CudaArrays
        """
        raise NotImplementedError
    
    def _process_block(
        self,
        block: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Process single block (to be implemented by subclasses).
        
        Args:
            block: Block data
            *args: Operation-specific arguments
            **kwargs: Operation-specific keyword arguments
        
        Returns:
            Processed block
        """
        raise NotImplementedError
    
    def _process_whole_array(
        self,
        array: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Process whole array (for FFT-like operations).
        
        Args:
            array: Whole array data
            *args: Operation-specific arguments
            **kwargs: Operation-specific keyword arguments
        
        Returns:
            Processed array
        """
        raise NotImplementedError
```

### 3. Specialized Vectorizer Subclasses

#### 3.1. ElementWiseVectorizer (`utils/cuda/elementwise_vectorizer.py`)

**Purpose:** Element-wise operations (addition, subtraction, multiplication, division, etc.)

**Operations:**
- Arithmetic: +, -, *, /, **
- Comparison: <, >, <=, >=, ==, !=
- Logical: &, |, ~
- Mathematical: sin, cos, exp, log, sqrt

**Use Cases:**
- Frequency conversions: l = π D ω
- Depth calculations: Δω = ω - ω_min
- Temperature conversions: ΔT = (Δω/ω_CMB) T_0
- Power spectrum: C_l ∝ ρ_Θ(ω(ℓ)) / ℓ²

#### 3.2. TransformVectorizer (`utils/cuda/transform_vectorizer.py`)

**Purpose:** Transform operations requiring whole arrays (FFT, spherical harmonics)

**Operations:**
- FFT / IFFT
- Spherical harmonic transforms
- Coordinate transformations
- Frequency domain operations

**Use Cases:**
- Spherical harmonic synthesis (Step 2.1)
- FFT-based correlation (Step 4.1)
- Frequency domain analysis (Step 3.1)

**Note:** Uses `whole_array=True` mode - cannot use block processing.

#### 3.3. ReductionVectorizer (`utils/cuda/reduction_vectorizer.py`)

**Purpose:** Reduction operations (sum, mean, max, min, etc.)

**Operations:**
- Sum, mean, std, var
- Max, min, argmax, argmin
- Any, all
- Custom reductions

**Use Cases:**
- Statistical calculations
- Peak finding
- Aggregation operations

#### 3.4. CorrelationVectorizer (`utils/cuda/correlation_vectorizer.py`)

**Purpose:** Correlation and convolution operations

**Operations:**
- Cross-correlation
- Auto-correlation
- Convolution
- Correlation functions

**Use Cases:**
- CMB-LSS correlation (Step 4.1)
- Correlation function calculations
- Signal correlation analysis

#### 3.5. GridVectorizer (`utils/cuda/grid_vectorizer.py`)

**Purpose:** Grid-based operations (local minima, gradients, etc.)

**Operations:**
- Local minimum/maximum detection
- Gradient calculations
- Curvature calculations
- Neighborhood operations

**Use Cases:**
- Node detection (Step 1.4)
- Grid-based processing
- Spatial analysis

---

## Implementation Plan

### Step 1: Array Model Implementation

**File:** `utils/cuda/array_model.py`

**Tasks:**
1. Implement `CudaArray` class with block-based storage
2. Implement swap functionality (CPU ↔ GPU)
3. Implement block access methods
4. Implement whole-array mode
5. Add memory management and error handling
6. Write unit tests

**Dependencies:** None (uses numpy, cupy for GPU)

### Step 2: Base Vectorizer Implementation

**File:** `utils/cuda/vectorizer.py`

**Tasks:**
1. Implement `BaseVectorizer` abstract base class
2. Implement block processing logic
3. Implement whole-array processing logic
4. Implement GPU detection and fallback
5. Add configuration and error handling
6. Write unit tests

**Dependencies:** `utils/cuda/array_model.py`

### Step 3: ElementWiseVectorizer Implementation

**File:** `utils/cuda/elementwise_vectorizer.py`

**Tasks:**
1. Implement `ElementWiseVectorizer` class
2. Implement element-wise arithmetic operations
3. Implement element-wise mathematical functions
4. Add GPU acceleration using CuPy
5. Add batch processing support
6. Write unit tests

**Dependencies:** `utils/cuda/vectorizer.py`

**Use Cases:**
- Phase 1: Frequency conversions, depth calculations
- Phase 2: Temperature conversions
- Phase 3: Power spectrum calculations

### Step 4: TransformVectorizer Implementation

**File:** `utils/cuda/transform_vectorizer.py`

**Tasks:**
1. Implement `TransformVectorizer` class
2. Implement FFT/IFFT operations (using CuPy FFT)
3. Implement spherical harmonic transforms
4. Add whole-array mode support
5. Add GPU acceleration
6. Write unit tests

**Dependencies:** `utils/cuda/vectorizer.py`, `utils/math/spherical_harmonics.py`

**Use Cases:**
- Phase 2: Spherical harmonic synthesis
- Phase 3: FFT-based spectrum analysis
- Phase 4: FFT-based correlation

### Step 5: ReductionVectorizer Implementation

**File:** `utils/cuda/reduction_vectorizer.py`

**Tasks:**
1. Implement `ReductionVectorizer` class
2. Implement standard reduction operations
3. Implement custom reduction operations
4. Add GPU acceleration
5. Write unit tests

**Dependencies:** `utils/cuda/vectorizer.py`

**Use Cases:**
- Phase 3: Statistical analysis
- Phase 4: Correlation statistics
- Phase 5-6: Report generation

### Step 6: CorrelationVectorizer Implementation

**File:** `utils/cuda/correlation_vectorizer.py`

**Tasks:**
1. Implement `CorrelationVectorizer` class
2. Implement cross-correlation operations
3. Implement correlation function calculations
4. Add GPU acceleration (using FFT-based correlation)
5. Write unit tests

**Dependencies:** `utils/cuda/transform_vectorizer.py`

**Use Cases:**
- Phase 4: CMB-LSS correlation (Step 4.1)
- Phase 4: Correlation function calculations

### Step 7: GridVectorizer Implementation

**File:** `utils/cuda/grid_vectorizer.py`

**Tasks:**
1. Implement `GridVectorizer` class
2. Implement local minimum/maximum detection
3. Implement gradient and curvature calculations
4. Implement neighborhood operations
5. Add GPU acceleration
6. Write unit tests

**Dependencies:** `utils/cuda/vectorizer.py`

**Use Cases:**
- Phase 1: Node detection (Step 1.4)
- Phase 1: Node classification
- Phase 2: Grid-based processing

### Step 8: Integration and Testing

**Tasks:**
1. Integration tests with actual CMB data
2. Performance benchmarking
3. Memory usage optimization
4. Documentation and examples
5. Update implementation plan with CUDA integration points

---

## Dependencies

### External Libraries

- **NumPy:** Base array operations (CPU)
- **CuPy:** GPU array operations (CUDA)
- **PyTorch (optional):** Alternative GPU backend
- **cupyx.scipy.fft:** FFT operations on GPU

### Internal Dependencies

- **Step 0.1** (Configuration) - For CUDA configuration settings
- **Step 0.2** (Utilities) - For mathematical operations (spherical harmonics)

---

## Configuration

### CUDA Configuration (`config/cmb_config.yaml`)

```yaml
cuda:
  enabled: true
  device: "cuda"  # or "cpu" for fallback
  block_size: null  # null = auto-detect
  use_whole_array_for_fft: true
  memory_limit_gb: 8  # GPU memory limit
  fallback_to_cpu: true  # Fallback to CPU if GPU fails
```

---

## Output Files

### Core Implementation Files

1. **`utils/cuda/array_model.py`** - CudaArray class
2. **`utils/cuda/vectorizer.py`** - BaseVectorizer abstract class
3. **`utils/cuda/elementwise_vectorizer.py`** - ElementWiseVectorizer
4. **`utils/cuda/transform_vectorizer.py`** - TransformVectorizer
5. **`utils/cuda/reduction_vectorizer.py`** - ReductionVectorizer
6. **`utils/cuda/correlation_vectorizer.py`** - CorrelationVectorizer
7. **`utils/cuda/grid_vectorizer.py`** - GridVectorizer
8. **`utils/cuda/__init__.py`** - Module exports

### Test Files

1. **`tests/unit/test_array_model.py`** - CudaArray tests
2. **`tests/unit/test_vectorizer.py`** - BaseVectorizer tests
3. **`tests/unit/test_elementwise_vectorizer.py`** - ElementWiseVectorizer tests
4. **`tests/unit/test_transform_vectorizer.py`** - TransformVectorizer tests
5. **`tests/unit/test_reduction_vectorizer.py`** - ReductionVectorizer tests
6. **`tests/unit/test_correlation_vectorizer.py`** - CorrelationVectorizer tests
7. **`tests/unit/test_grid_vectorizer.py`** - GridVectorizer tests
8. **`tests/integration/test_cuda_integration.py`** - Integration tests

### Documentation Files

1. **`docs/api/cuda.md`** - CUDA API documentation
2. **`docs/examples/cuda_usage.md`** - Usage examples

---

## Tests

### Unit Tests

1. **Array Model Tests**
   - Block creation and access
   - Swap operations (CPU ↔ GPU)
   - Whole-array mode
   - Memory management

2. **Vectorizer Base Tests**
   - Block processing logic
   - Whole-array processing logic
   - GPU detection and fallback
   - Error handling

3. **ElementWiseVectorizer Tests**
   - Element-wise operations
   - Batch processing
   - GPU acceleration
   - Performance validation

4. **TransformVectorizer Tests**
   - FFT operations
   - Spherical harmonic transforms
   - Whole-array mode
   - GPU acceleration

5. **ReductionVectorizer Tests**
   - Reduction operations
   - GPU acceleration
   - Performance validation

6. **CorrelationVectorizer Tests**
   - Correlation operations
   - FFT-based correlation
   - GPU acceleration

7. **GridVectorizer Tests**
   - Local minimum detection
   - Gradient calculations
   - GPU acceleration

### Integration Tests

1. **End-to-End CUDA Pipeline**
   - Load data → Process with CUDA → Validate results
   - Compare CPU vs GPU results
   - Performance benchmarking

2. **Real CMB Data Processing**
   - Process actual CMB data with CUDA
   - Validate correctness
   - Measure performance improvement

---

## Implementation Notes

### Priority Order (from cuda.mdc)

1. **Block-based processing** - Highest priority
   - All operations should support block processing
   - Memory-efficient for large arrays

2. **Vectorization** - High priority
   - All operations should be vectorized
   - Use NumPy/CuPy vectorized operations

3. **Batching** - Medium priority
   - Support batch processing for multiple arrays
   - Improve GPU utilization

4. **CUDA acceleration** - Based on above
   - Use GPU when available
   - Fallback to CPU if GPU unavailable

### Memory Management

- **Block size:** Auto-detect based on available GPU memory
- **Swap strategy:** Swap blocks as needed, not entire arrays
- **Memory limits:** Respect GPU memory limits
- **Cleanup:** Properly free GPU memory after operations

### Error Handling

- **GPU unavailable:** Fallback to CPU automatically
- **Memory errors:** Handle out-of-memory gracefully
- **Type errors:** Validate array types and shapes
- **Device errors:** Handle CUDA device errors

### Performance Optimization

- **Kernel fusion:** Combine multiple operations when possible
- **Memory coalescing:** Optimize memory access patterns
- **Stream processing:** Use CUDA streams for parallel operations
- **Profiling:** Provide profiling tools for optimization

---

## Related Steps

- **Step 0.1** (Configuration) - CUDA configuration
- **Step 0.2** (Utilities) - Mathematical operations
- **All Phase 1-6 steps** - Will use CUDA acceleration

---

## Success Criteria

1. ✅ Array model supports block-based processing and swap
2. ✅ Base vectorizer class provides foundation for all operations
3. ✅ All specialized vectorizers implemented and tested
4. ✅ GPU acceleration provides significant speedup (>5x) for large arrays
5. ✅ CPU fallback works correctly when GPU unavailable
6. ✅ Memory usage is efficient (block-based processing)
7. ✅ Whole-array mode works for FFT operations
8. ✅ Integration with existing code is seamless

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com
