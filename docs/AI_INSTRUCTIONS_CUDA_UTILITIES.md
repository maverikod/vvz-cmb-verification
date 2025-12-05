# AI Instructions: CUDA Utilities Priority Usage

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

---

## 🚨 КРИТИЧЕСКИ ВАЖНО: ПРИОРИТЕТ ИСПОЛЬЗОВАНИЯ

**ВСЕГДА используй утилиты из `utils/cuda/` вместо прямых операций с numpy массивами.**

### Приоритеты (в порядке важности):

1. **Блочная обработка (блочная обработка)** - ВЫСШИЙ ПРИОРИТЕТ
2. **Векторизация (векторизация)** - ВЫСОКИЙ ПРИОРИТЕТ  
3. **Батчинг (батчинг)** - СРЕДНИЙ ПРИОРИТЕТ
4. **CUDA ускорение** - на основе вышеперечисленного

---

## ❌ ЗАПРЕЩЕНО

### НЕ используй прямые numpy операции:

```python
# ❌ НЕПРАВИЛЬНО:
import numpy as np
result = np.add(array1, array2)
result = array1 + array2
result = np.sum(array)
result = np.mean(array)
result = np.diff(array)
result = array[1:] - array[:-1]
```

### НЕ используй прямые операции с массивами:

```python
# ❌ НЕПРАВИЛЬНО:
derivatives = np.zeros(n)
derivatives[1:-1] = (values[2:] - values[:-2]) / (times[2:] - times[:-2])
gap_mask = intervals > threshold
```

---

## ✅ ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ

### 1. CudaArray для всех массивов

**ВСЕГДА оборачивай numpy массивы в CudaArray:**

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import CudaArray

# Создание CudaArray
times_cuda = CudaArray(times, device="cpu")
values_cuda = CudaArray(values, device="cpu")

# Получение numpy только когда необходимо для срезов
times_np = times_cuda.to_numpy()
times_forward = CudaArray(times_np[2:], device="cpu")
```

**Методы CudaArray:**
- `CudaArray(data, block_size=None, device="cpu")` - создание
- `.to_numpy()` - конвертация в numpy (только для срезов)
- `.use_whole_array()` - получение целого массива (для FFT)
- `.swap_to_gpu()` - перенос на GPU
- `.swap_to_cpu()` - перенос на CPU
- `.get_block(block_idx)` - получение блока
- `.process_blocks(operation, use_gpu=True)` - обработка блоками

---

### 2. ElementWiseVectorizer для элементных операций

**ВСЕГДА используй ElementWiseVectorizer для арифметики и математики:**

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import ElementWiseVectorizer

elem_vec = ElementWiseVectorizer(use_gpu=True)

# Арифметика
result = elem_vec.add(array_cuda, 10.0)
result = elem_vec.subtract(array1_cuda, array2_cuda)
result = elem_vec.multiply(array_cuda, 2.0)
result = elem_vec.divide(array1_cuda, array2_cuda)
result = elem_vec.power(array_cuda, 2.0)

# Математика
result = elem_vec.sin(array_cuda)
result = elem_vec.cos(array_cuda)
result = elem_vec.exp(array_cuda)
result = elem_vec.log(array_cuda)
result = elem_vec.sqrt(array_cuda)
result = elem_vec.abs(array_cuda)

# Сравнения
mask = elem_vec.vectorize_operation(array_cuda, "greater", threshold)
mask = elem_vec.vectorize_operation(array_cuda, "less_equal", 0.0)
```

**Поддерживаемые операции:**
- Арифметика: `add`, `subtract`, `multiply`, `divide`, `power`
- Математика: `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`, `sign`
- Сравнение: `less`, `greater`, `less_equal`, `greater_equal`, `equal`, `not_equal`

---

### 3. ReductionVectorizer для редукций

**ВСЕГДА используй ReductionVectorizer для сумм, средних, максимумов и т.д.:**

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import ReductionVectorizer

reduction_vec = ReductionVectorizer(use_gpu=True)

# Редукции
sum_result = reduction_vec.vectorize_reduction(array_cuda, "sum")
mean_result = reduction_vec.vectorize_reduction(array_cuda, "mean")
std_result = reduction_vec.vectorize_reduction(array_cuda, "std")
max_result = reduction_vec.vectorize_reduction(array_cuda, "max")
min_result = reduction_vec.vectorize_reduction(array_cuda, "min")
any_result = reduction_vec.vectorize_reduction(array_cuda, "any")
all_result = reduction_vec.vectorize_reduction(array_cuda, "all")
```

**Поддерживаемые редукции:**
- Стандартные: `sum`, `mean`, `std`, `var`, `max`, `min`, `argmax`, `argmin`
- Логические: `any`, `all`

---

### 4. TransformVectorizer для трансформаций

**ВСЕГДА используй TransformVectorizer для FFT и сферических гармоник:**

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import TransformVectorizer

transform_vec = TransformVectorizer(use_gpu=True)

# FFT операции
fft_result = transform_vec.vectorize_transform(array_cuda, "fft")
ifft_result = transform_vec.vectorize_transform(array_cuda, "ifft")
rfft_result = transform_vec.vectorize_transform(array_cuda, "rfft")
```

---

### 5. GridVectorizer для сеточных операций

**ВСЕГДА используй GridVectorizer для градиентов, минимумов и т.д.:**

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import GridVectorizer

grid_vec = GridVectorizer(use_gpu=True)

# Сеточные операции
minima = grid_vec.vectorize_grid_operation(array_cuda, "local_minima")
gradient = grid_vec.vectorize_grid_operation(array_cuda, "gradient")
laplacian = grid_vec.vectorize_grid_operation(array_cuda, "laplacian")
```

---

### 6. CorrelationVectorizer для корреляций

**ВСЕГДА используй CorrelationVectorizer для корреляций:**

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import CorrelationVectorizer

corr_vec = CorrelationVectorizer(use_gpu=True)

# Корреляции
correlation = corr_vec.vectorize_correlation(array1_cuda, array2_cuda, method="fft")
```

---

### 7. Батчинг для множественных массивов

**ВСЕГДА используй batch() для обработки нескольких массивов:**

```python
# ✅ ПРАВИЛЬНО:
arrays = [
    CudaArray(data1, device="cpu"),
    CudaArray(data2, device="cpu"),
    CudaArray(data3, device="cpu")
]

# Батч обработка
elem_vec = ElementWiseVectorizer(use_gpu=True)
results = elem_vec.batch(arrays, "multiply", 2.0)
```

---

## 📋 ШАБЛОН ПРАВИЛЬНОГО КОДА

### Пример: Вычисление производной

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import CudaArray, ElementWiseVectorizer

def calculate_derivative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Calculate derivative using CUDA utilities."""
    # 1. Обернуть в CudaArray
    times_cuda = CudaArray(times, device="cpu")
    values_cuda = CudaArray(values, device="cpu")
    
    # 2. Получить numpy только для срезов
    times_np = times_cuda.to_numpy()
    values_np = values_cuda.to_numpy()
    
    # 3. Создать CudaArray для срезов
    times_forward = CudaArray(times_np[2:], device="cpu")
    times_backward = CudaArray(times_np[:-2], device="cpu")
    values_forward = CudaArray(values_np[2:], device="cpu")
    values_backward = CudaArray(values_np[:-2], device="cpu")
    
    # 4. Использовать ElementWiseVectorizer для операций
    elem_vec = ElementWiseVectorizer(use_gpu=True)
    dt_cuda = elem_vec.subtract(times_forward, times_backward)
    dv_cuda = elem_vec.subtract(values_forward, values_backward)
    
    # 5. Деление через ElementWiseVectorizer
    derivatives_cuda = elem_vec.divide(dv_cuda, dt_cuda)
    
    # 6. Конвертировать результат в numpy
    derivatives = derivatives_cuda.to_numpy()
    
    # 7. Очистка GPU памяти
    if times_forward.device == "cuda":
        times_forward.swap_to_cpu()
    if times_backward.device == "cuda":
        times_backward.swap_to_cpu()
    # ... и т.д. для всех CudaArray
    
    return derivatives
```

### Пример: Проверка gaps

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import CudaArray, ElementWiseVectorizer, ReductionVectorizer

def check_gaps(times: np.ndarray, max_gap_ratio: float = 5.0) -> List[Tuple[float, float]]:
    """Check for gaps using CUDA utilities."""
    # 1. Обернуть в CudaArray
    times_cuda = CudaArray(times, device="cpu")
    times_np = times_cuda.to_numpy()
    
    # 2. Вычислить интервалы (np.diff для срезов, затем CudaArray)
    intervals_np = np.diff(times_np)
    intervals_cuda = CudaArray(intervals_np, device="cpu")
    
    # 3. Вычислить медиану (требует сортировку, используем numpy)
    median_interval = float(np.median(intervals_np))
    
    # 4. Вычислить threshold
    threshold = max_gap_ratio * median_interval
    
    # 5. Использовать ElementWiseVectorizer для сравнения
    elem_vec = ElementWiseVectorizer(use_gpu=True)
    threshold_cuda = CudaArray(np.array([threshold]), device="cpu")
    gap_mask_cuda = elem_vec.vectorize_operation(
        intervals_cuda, "greater", threshold_cuda.to_numpy()[0]
    )
    gap_mask = gap_mask_cuda.to_numpy()
    
    # 6. Очистка GPU памяти
    if intervals_cuda.device == "cuda":
        intervals_cuda.swap_to_cpu()
    if gap_mask_cuda.device == "cuda":
        gap_mask_cuda.swap_to_cpu()
    if threshold_cuda.device == "cuda":
        threshold_cuda.swap_to_cpu()
    if times_cuda.device == "cuda":
        times_cuda.swap_to_cpu()
    
    # 7. Найти gaps
    gap_indices = np.where(gap_mask)[0]
    gaps = [(float(times_np[i]), float(times_np[i + 1])) for i in gap_indices]
    
    return gaps
```

### Пример: Статистика

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import CudaArray, ReductionVectorizer

def compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """Compute statistics using CUDA utilities."""
    # 1. Обернуть в CudaArray
    values_cuda = CudaArray(values, device="cpu")
    
    # 2. Использовать ReductionVectorizer для всех редукций
    reduction_vec = ReductionVectorizer(use_gpu=True)
    
    mean_result = reduction_vec.vectorize_reduction(values_cuda, "mean")
    std_result = reduction_vec.vectorize_reduction(values_cuda, "std")
    min_result = reduction_vec.vectorize_reduction(values_cuda, "min")
    max_result = reduction_vec.vectorize_reduction(values_cuda, "max")
    
    # 3. Конвертировать результаты
    from utils.cuda.array_model import CudaArray as CA
    def _to_float(value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, CA):
            return float(value.to_numpy().item())
        return float(value)
    
    # 4. Очистка GPU памяти
    if values_cuda.device == "cuda":
        values_cuda.swap_to_cpu()
    
    return {
        "mean": _to_float(mean_result),
        "std": _to_float(std_result),
        "min": _to_float(min_result),
        "max": _to_float(max_result),
    }
```

---

## 🔄 ПРАВИЛА ОЧИСТКИ GPU ПАМЯТИ

**ВСЕГДА очищай GPU память после использования:**

```python
# ✅ ПРАВИЛЬНО:
if array_cuda.device == "cuda":
    array_cuda.swap_to_cpu()
```

**Проверяй device перед очисткой для всех CudaArray объектов.**

---

## 📝 ЧЕКЛИСТ ПЕРЕД НАПИСАНИЕМ КОДА

Перед написанием кода, который работает с массивами:

- [ ] Использую ли я `CudaArray` для всех массивов?
- [ ] Использую ли я `ElementWiseVectorizer` для арифметики и математики?
- [ ] Использую ли я `ReductionVectorizer` для сумм, средних, максимумов?
- [ ] Использую ли я `TransformVectorizer` для FFT операций?
- [ ] Использую ли я `GridVectorizer` для градиентов и минимумов?
- [ ] Использую ли я `CorrelationVectorizer` для корреляций?
- [ ] Использую ли я `batch()` для обработки нескольких массивов?
- [ ] Очищаю ли я GPU память после использования?
- [ ] Нет ли прямых numpy операций вместо утилит?
- [ ] Нет ли прямых операций с массивами (срезы, индексация) без CudaArray?

---

## 🎯 ПРИМЕРЫ КОНВЕРТАЦИИ

### Конвертация из неправильного кода:

```python
# ❌ НЕПРАВИЛЬНО:
def calculate_derivative(times, values):
    n = len(times)
    derivatives = np.zeros(n)
    derivatives[1:-1] = (values[2:] - values[:-2]) / (times[2:] - times[:-2])
    return derivatives
```

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import CudaArray, ElementWiseVectorizer

def calculate_derivative(times, values):
    # Обернуть в CudaArray
    times_cuda = CudaArray(times, device="cpu")
    values_cuda = CudaArray(values, device="cpu")
    
    # Получить numpy для срезов
    times_np = times_cuda.to_numpy()
    values_np = values_cuda.to_numpy()
    
    # Создать CudaArray для срезов
    times_forward = CudaArray(times_np[2:], device="cpu")
    times_backward = CudaArray(times_np[:-2], device="cpu")
    values_forward = CudaArray(values_np[2:], device="cpu")
    values_backward = CudaArray(values_np[:-2], device="cpu")
    
    # Использовать ElementWiseVectorizer
    elem_vec = ElementWiseVectorizer(use_gpu=True)
    dt_cuda = elem_vec.subtract(times_forward, times_backward)
    dv_cuda = elem_vec.subtract(values_forward, values_backward)
    derivatives_cuda = elem_vec.divide(dv_cuda, dt_cuda)
    
    # Очистка
    for arr in [times_forward, times_backward, values_forward, values_backward, dt_cuda, dv_cuda]:
        if arr.device == "cuda":
            arr.swap_to_cpu()
    
    return derivatives_cuda.to_numpy()
```

---

## 📚 ИМПОРТЫ

**ВСЕГДА используй правильные импорты:**

```python
# ✅ ПРАВИЛЬНО:
from utils.cuda import (
    CudaArray,
    ElementWiseVectorizer,
    ReductionVectorizer,
    TransformVectorizer,
    GridVectorizer,
    CorrelationVectorizer,
)
```

---

## ⚠️ ИСКЛЮЧЕНИЯ

**Только в следующих случаях можно использовать numpy напрямую:**

1. **Срезы массивов** - для создания срезов используй `to_numpy()`, затем создавай новый `CudaArray`
2. **Сортировка** - `np.sort()`, `np.argsort()` (нет CUDA аналогов)
3. **Медиана** - `np.median()` (требует сортировку)
4. **Индексация** - `np.where()`, `np.argwhere()` (для поиска индексов)
5. **Создание массивов** - `np.zeros()`, `np.ones()`, `np.array()` (затем оборачивай в `CudaArray`)

**НО:** После любой numpy операции сразу оборачивай результат в `CudaArray`!

---

## 🚀 ПРОИЗВОДИТЕЛЬНОСТЬ

**Пороги для использования CUDA:**

- Массивы > 10,000 элементов: используй CUDA
- Массивы < 10,000 элементов: можно использовать CPU (но все равно через утилиты для консистентности)

**Всегда используй утилиты, даже для маленьких массивов - это обеспечивает консистентность кода.**

---

**ПОМНИ: Утилиты из `utils/cuda/` - это ОБЯЗАТЕЛЬНЫЙ стандарт проекта. Всегда используй их вместо прямых numpy операций!**

