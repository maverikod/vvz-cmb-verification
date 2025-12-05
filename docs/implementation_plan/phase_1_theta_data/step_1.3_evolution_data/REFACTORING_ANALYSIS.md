# Анализ использования новых классов (Phase 1, Steps 1.1-1.3)

**Дата анализа:** 2024  
**Охват:** Phase 1 - Θ-Field Data Processing (Steps 1.1, 1.2, 1.3)

---

## 📋 Созданные классы и утилиты

### Step 1.1: Θ-Field Data Loader
**Модуль:** `cmb/theta_data_loader.py`

1. **`ThetaFrequencySpectrum`** (dataclass)
   - Хранит частотный спектр ρ_Θ(ω,t)
   - Атрибуты: `frequencies`, `times`, `spectrum`, `metadata`
   - Использование: `from cmb.theta_data_loader import ThetaFrequencySpectrum`

2. **`ThetaEvolution`** (dataclass)
   - Хранит данные временной эволюции ω_min(t), ω_macro(t)
   - Атрибуты: `times`, `omega_min`, `omega_macro`, `metadata`
   - Использование: `from cmb.theta_data_loader import ThetaEvolution`

3. **Функции загрузки:**
   - `load_frequency_spectrum(data_path=None) -> ThetaFrequencySpectrum`
   - `load_evolution_data(data_path=None) -> ThetaEvolution`
   - `validate_frequency_spectrum(spectrum: ThetaFrequencySpectrum) -> bool`
   - `validate_evolution_data(evolution: ThetaEvolution) -> bool`

### Step 1.2: Θ-Node Data Processing
**Модуль:** `cmb/theta_node_processor.py`

1. **`ThetaNodeData`** (dataclass)
   - Хранит данные узлов Θ-поля
   - Атрибуты: `positions`, `scales`, `depths`, `temperatures`, `metadata`
   - Использование: `from cmb.theta_node_processor import ThetaNodeData`

2. **Функции обработки:**
   - `process_node_data(geometry_path=None, depth_path=None) -> ThetaNodeData`
   - `map_depth_to_temperature(depths: np.ndarray, config=None) -> np.ndarray`

**Модуль:** `cmb/theta_node_loader.py`

1. **Функции загрузки:**
   - `load_node_geometry(data_path=None) -> Tuple[np.ndarray, np.ndarray]`
   - `load_node_depths(data_path=None) -> np.ndarray`

### Step 1.3: Θ-Field Evolution Data Processing
**Модуль:** `cmb/theta_evolution_processor.py`

1. **`ThetaEvolutionProcessor`** (class)
   - Обрабатывает данные временной эволюции
   - Методы:
     - `process()` - создание интерполяторов
     - `get_omega_min(time: float) -> float`
     - `get_omega_macro(time: float) -> float`
     - `get_evolution_rate_min(time: float) -> float`
     - `get_evolution_rate_macro(time: float) -> float`
     - `validate_against_config() -> bool`
     - `verify_time_array_completeness(expected_interval=None) -> Dict`
     - `generate_quality_report() -> Dict`
   - Использование: `from cmb.theta_evolution_processor import ThetaEvolutionProcessor`

2. **Функция-фасад:**
   - `process_evolution_data(evolution: ThetaEvolution) -> ThetaEvolutionProcessor`

### Утилиты (Phase 0)

**Модуль:** `utils/io/data_loader.py`
- `load_csv_data(file_path: Path) -> Dict[str, np.ndarray]`
- `load_json_data(file_path: Path) -> Dict[str, Any]`
- `load_healpix_map(file_path: Path) -> np.ndarray`

**Модуль:** `utils/io/data_index_loader.py`
- `DataIndex` (class) - загрузка и поиск файлов через `data/in/data_index.yaml`
  - `DataIndex.load(index_path=None) -> DataIndex`
  - `get_files_by_category(category: str) -> List[Dict]`
  - `get_file_path(category: str, file_name: str) -> Optional[Path]`

**Модуль:** `utils/io/data_saver.py`
- `save_csv_data(data: Dict, file_path: Path) -> None`
- `save_json_data(data: Dict, file_path: Path) -> None`
- `save_healpix_map(map_data: np.ndarray, file_path: Path) -> None`

**Модуль:** `utils/math/frequency_conversion.py`
- `frequency_to_multipole(frequency: float, D: float) -> float`
- `multipole_to_frequency(multipole: float, D: float) -> float`

**Модуль:** `utils/math/spherical_harmonics.py`
- `synthesize_map(alm: np.ndarray, nside: int) -> np.ndarray`
- `decompose_map(map_data: np.ndarray, l_max: int) -> np.ndarray`

---

## 🔍 Места, где нужно использовать новые классы

### ✅ Правильно используемые места

1. **`docs/implementation_plan/phase_2_cmb_reconstruction/step_2.1_reconstruction_core/cmb_map_reconstructor.py`**
   - ✅ Использует: `ThetaFrequencySpectrum`, `ThetaEvolution`, `ThetaNodeData`
   - ✅ Импорты корректны

2. **`docs/implementation_plan/phase_3_power_spectrum/step_3.1_spectrum_calculation/power_spectrum.py`**
   - ✅ Использует: `ThetaFrequencySpectrum`, `ThetaEvolution`, `ThetaEvolutionProcessor`
   - ✅ Импорты корректны

3. **`docs/implementation_plan/phase_3_power_spectrum/step_3.2_subpeaks_analysis/subpeaks_analyzer.py`**
   - ✅ Использует: `ThetaEvolutionProcessor`
   - ✅ Импорты корректны

4. **`docs/implementation_plan/phase_5_act_spt_predictions/step_5.1_highl_peak/highl_peak_predictor.py`**
   - ✅ Использует: `ThetaEvolutionProcessor`
   - ✅ Импорты корректны

5. **`docs/implementation_plan/phase_2_cmb_reconstruction/step_2.3_node_mapping/node_to_cmb_mapper.py`**
   - ✅ Использует: `ThetaNodeData`
   - ✅ Импорты корректны

### ⚠️ Места, требующие рефакторинга

#### 1. **`docs/implementation_plan/phase_1_theta_data/step_1.4_node_map_generation/theta_node_map.py`**

**Текущее состояние:**
- Использует только `ThetaFrequencySpectrum`
- НЕ использует `ThetaNodeData` для хранения результатов
- НЕ использует `process_node_data()` для обработки узлов

**Что нужно изменить:**
```python
# ВМЕСТО:
class ThetaNodeMapGenerator:
    def __init__(self, omega_field: np.ndarray, ...):
        # Прямая работа с numpy массивами
        
# ДОЛЖНО БЫТЬ:
from cmb.theta_node_processor import ThetaNodeData, process_node_data

class ThetaNodeMapGenerator:
    def __init__(
        self,
        frequency_spectrum: ThetaFrequencySpectrum,  # Использовать класс
        config: Optional[Config] = None,
    ):
        # Использовать ThetaFrequencySpectrum для получения omega_field
        # Результат должен возвращать ThetaNodeData
```

**Конкретные изменения:**
- Метод `generate_map()` должен возвращать `ThetaNodeData` вместо `ThetaNodeMap`
- Использовать `process_node_data()` для обработки геометрии и глубин
- Использовать `map_depth_to_temperature()` для конвертации глубин

---

#### 2. **`docs/implementation_plan/phase_2_cmb_reconstruction/step_2.2_map_validation/map_validator.py`**

**Текущее состояние:**
- Использует прямой импорт `load_healpix_map` (правильно)
- НО: может использовать `DataIndex` для поиска файлов наблюдений

**Что нужно изменить:**
```python
# ВМЕСТО:
def load_observed_map(self) -> None:
    # Прямой путь к файлу
    self.observed_map = load_healpix_map(self.observed_map_path)

# ДОЛЖНО БЫТЬ:
from utils.io.data_index_loader import DataIndex

def load_observed_map(self) -> None:
    # Использовать DataIndex для поиска файла
    if self.observed_map_path is None:
        data_index = DataIndex.load()
        act_files = data_index.get_files_by_category("act_observations")
        # Найти файл ACT DR6.02
        ...
    self.observed_map = load_healpix_map(self.observed_map_path)
```

---

#### 3. **`docs/implementation_plan/phase_4_cmb_lss_correlation/step_4.1_correlation_core/cmb_lss_correlator.py`**

**Текущее состояние:**
- Неизвестно, использует ли новые классы

**Что нужно проверить:**
- Должен использовать `ThetaFrequencySpectrum` для доступа к спектру
- Должен использовать `ThetaEvolutionProcessor` для временных параметров
- Должен использовать `DataIndex` для поиска LSS данных

---

#### 4. **`docs/implementation_plan/phase_4_cmb_lss_correlation/step_4.3_node_lss_mapping/node_lss_mapper.py`**

**Текущее состояние:**
- Использует `ThetaNodeData` (правильно)

**Что нужно проверить:**
- Должен использовать `DataIndex` для поиска LSS данных
- Должен использовать `utils.io.data_loader` для загрузки LSS файлов

---

#### 5. **`docs/implementation_plan/phase_5_act_spt_predictions/step_5.2_frequency_invariance/frequency_invariance.py`**

**Текущее состояние:**
- Неизвестно, использует ли `ThetaEvolutionProcessor`

**Что нужно проверить:**
- Должен использовать `ThetaEvolutionProcessor` для получения ω_min(t), ω_macro(t)
- Должен использовать `get_evolution_rate_min()`, `get_evolution_rate_macro()`

---

#### 6. **`docs/implementation_plan/phase_5_act_spt_predictions/step_5.3_predictions_report/predictions_report.py`**

**Текущее состояние:**
- Неизвестно, использует ли новые классы

**Что нужно проверить:**
- Должен использовать `utils.io.data_saver` для сохранения отчетов
- Должен использовать `DataIndex` для поиска данных для сравнения

---

#### 7. **`docs/implementation_plan/phase_6_chain_verification/step_6.1_cluster_plateau/cluster_plateau_analyzer.py`**

**Текущее состояние:**
- Неизвестно, использует ли новые классы

**Что нужно проверить:**
- Должен использовать `ThetaNodeData` для работы с узлами
- Должен использовать `DataIndex` для поиска данных кластеров

---

#### 8. **`docs/implementation_plan/phase_6_chain_verification/step_6.2_galaxy_distribution/galaxy_distribution_analyzer.py`**

**Текущее состояние:**
- Неизвестно, использует ли новые классы

**Что нужно проверить:**
- Должен использовать `DataIndex` для поиска данных распределения галактик
- Должен использовать `utils.io.data_loader` для загрузки данных

---

#### 9. **`docs/implementation_plan/phase_6_chain_verification/step_6.3_chain_report/chain_verifier.py`**

**Текущее состояние:**
- Неизвестно, использует ли новые классы

**Что нужно проверить:**
- Должен использовать `utils.io.data_saver` для сохранения отчетов
- Должен использовать `DataIndex` для поиска всех необходимых данных

---

## 📝 Общие рекомендации

### ❌ ЗАПРЕЩЕНО:

1. **Прямая загрузка CSV/JSON файлов без утилит:**
   ```python
   # ❌ НЕПРАВИЛЬНО:
   import pandas as pd
   data = pd.read_csv("data/theta/spectrum.csv")
   
   # ✅ ПРАВИЛЬНО:
   from utils.io.data_loader import load_csv_data
   data = load_csv_data(Path("data/theta/spectrum.csv"))
   ```

2. **Прямая работа с numpy массивами вместо классов:**
   ```python
   # ❌ НЕПРАВИЛЬНО:
   times = np.load("data/theta/evolution.npy")
   omega_min = np.load("data/theta/omega_min.npy")
   
   # ✅ ПРАВИЛЬНО:
   from cmb.theta_data_loader import load_evolution_data
   evolution = load_evolution_data()
   times = evolution.times
   omega_min = evolution.omega_min
   ```

3. **Хардкод путей к файлам:**
   ```python
   # ❌ НЕПРАВИЛЬНО:
   file_path = Path("data/theta/spectrum.csv")
   
   # ✅ ПРАВИЛЬНО:
   from utils.io.data_index_loader import DataIndex
   data_index = DataIndex.load()
   file_path = data_index.get_file_path("theta_field_data", "spectrum.csv")
   ```

4. **Прямая работа с временной эволюцией без процессора:**
   ```python
   # ❌ НЕПРАВИЛЬНО:
   evolution = load_evolution_data()
   omega_min_t = np.interp(t, evolution.times, evolution.omega_min)
   
   # ✅ ПРАВИЛЬНО:
   from cmb.theta_evolution_processor import process_evolution_data
   processor = process_evolution_data(evolution)
   processor.process()
   omega_min_t = processor.get_omega_min(t)
   ```

### ✅ ОБЯЗАТЕЛЬНО:

1. **Использовать классы данных:**
   - `ThetaFrequencySpectrum` для частотного спектра
   - `ThetaEvolution` для данных эволюции
   - `ThetaNodeData` для данных узлов

2. **Использовать процессоры:**
   - `ThetaEvolutionProcessor` для работы с временной эволюцией
   - `process_node_data()` для обработки узлов

3. **Использовать утилиты загрузки:**
   - `utils.io.data_loader` для загрузки файлов
   - `utils.io.data_index_loader.DataIndex` для поиска файлов
   - `utils.io.data_saver` для сохранения результатов

4. **Использовать утилиты математики:**
   - `utils.math.frequency_conversion` для конвертации частот
   - `utils.math.spherical_harmonics` для работы со сферическими гармониками

---

## 🎯 Приоритеты рефакторинга

### Высокий приоритет:
1. `step_1.4_node_map_generation/theta_node_map.py` - должен возвращать `ThetaNodeData`
2. Все файлы, которые загружают данные напрямую без `DataIndex`

### Средний приоритет:
3. Файлы, которые работают с временной эволюцией без `ThetaEvolutionProcessor`
4. Файлы, которые сохраняют данные без `utils.io.data_saver`

### Низкий приоритет:
5. Файлы, которые уже используют классы, но могут использовать дополнительные утилиты

---

**Author:** Vasiliy Zdanovskiy  
**Email:** vasilyvz@gmail.com

