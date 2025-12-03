# ML-DFT Feedback System API Documentation

## Overview

ML-DFT Feedback System은 머신러닝과 밀도범함수이론(DFT)을 결합한 재료 역설계 프레임워크입니다.
목표 물성을 입력하면 해당 물성을 가진 구조를 자동으로 탐색합니다.

## Architecture

```
Target Properties
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Mutation   │────▶│     ML       │────▶│    DFT      │
│  Generator  │     │  Predictor   │     │  Validator  │
└─────────────┘     └──────────────┘     └─────────────┘
       ▲                                        │
       │           Feedback Loop                │
       └────────────────────────────────────────┘
```

---

## Core Module (`core/`)

### Structure

구조 데이터를 나타내는 기본 데이터 클래스입니다.

```python
from core.interfaces import Structure

structure = Structure(
    lattice_vectors=np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]),
    positions=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
    atomic_numbers=[14, 14],
    species=["Si", "Si"],
    structure_id="si_001"
)
```

**Parameters:**
- `lattice_vectors` (np.ndarray): 3x3 격자 벡터 (Angstrom)
- `positions` (np.ndarray): Nx3 분율 좌표
- `atomic_numbers` (List[int]): 원자 번호 리스트
- `species` (List[str]): 원소 기호 리스트
- `structure_id` (str, optional): 고유 구조 ID
- `properties` (Dict, optional): 추가 속성

### MLDFTSystem

시스템의 메인 클래스입니다.

```python
from core.system import MLDFTSystem
from config.settings import load_config

config = load_config("config.yaml")
system = MLDFTSystem(config)

# 최적화 실행
result = await system.run_optimization(
    initial_structure=structure,
    target_properties={"band_gap": 1.5},
    max_iterations=100
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `run_optimization()` | 구조 최적화 실행 |
| `predict_properties()` | ML 모델로 물성 예측 |
| `validate_structure()` | DFT 검증 수행 |
| `generate_mutations()` | 구조 변이 생성 |

---

## Mutation Module (`models/mutation/`)

### MutationGenerator

구조 변이를 생성합니다.

```python
from models.mutation.generator import MutationGenerator, MutationType

generator = MutationGenerator()

# 단일 변이 생성
result = generator.generate_mutation(
    structure=structure,
    mutation_type=MutationType.SUBSTITUTION
)

# 다중 변이 생성
results = generator.generate_mutations(
    structure=structure,
    n_mutations=10
)
```

**MutationType:**
- `SUBSTITUTION`: 원소 치환
- `DISTORTION`: 격자 변형
- `VACANCY`: 빈자리 생성
- `INTERSTITIAL`: 격자간 원자 삽입
- `SWAP`: 원자 위치 교환

### MutationValidator

변이 구조의 유효성을 검증합니다.

```python
from models.mutation.validator import MutationValidator

validator = MutationValidator()

# 변이 검증
validation = validator.validate(mutation_result)
if validation.is_valid:
    print("Valid mutation")
else:
    print(f"Invalid: {validation.errors}")
```

---

## ML Module (`models/ml/`)

### PropertyPredictor

GNN 기반 물성 예측 모델입니다.

```python
from models.ml.predictor import PropertyPredictor

predictor = PropertyPredictor(
    hidden_dim=64,
    n_layers=3,
    target_properties=["band_gap", "formation_energy"]
)

# 물성 예측
prediction = predictor.predict(structure)
print(f"Band gap: {prediction.properties['band_gap']}")
print(f"Uncertainty: {prediction.uncertainty}")
```

### ModelTrainer

ML 모델 학습 관리자입니다.

```python
from models.ml.trainer import ModelTrainer

trainer = ModelTrainer(
    model=predictor,
    learning_rate=0.001,
    batch_size=32
)

# 모델 학습
history = trainer.train(
    train_data=train_dataset,
    val_data=val_dataset,
    epochs=100
)
```

---

## DFT Module (`models/dft/`)

### DFTCalculator

DFT 계산을 관리합니다.

```python
from models.dft.calculator import DFTCalculator, DFTCode

calculator = DFTCalculator(dft_code=DFTCode.VASP)

# 계산 실행
result = await calculator.calculate(
    structure=structure,
    calculation_type="scf",
    parameters={"encut": 500, "kpoints": [4, 4, 4]}
)
```

**Supported DFT Codes:**
- `VASP`
- `QE` (Quantum ESPRESSO)
- `SIESTA`

### DFTOutputParser

DFT 출력 파일을 파싱합니다.

```python
from models.dft.parser import DFTOutputParser

parser = DFTOutputParser(dft_code="vasp")
results = parser.parse_output(calc_dir=Path("./vasp_calc"))

print(f"Total energy: {results['energies'][-1]} eV")
print(f"Band gap: {results.get('band_gap')} eV")
```

---

## Validation Module (`models/validation/`)

### StructureChecker

구조 유효성을 검사합니다.

```python
from models.validation.checker import StructureChecker

checker = StructureChecker()

# 기하학적 검사
geometry_check = checker.check_geometry(structure)

# 거리 검사
distance_check = checker.check_distances(structure)

# 산화수 검사
oxidation_check = checker.check_oxidation_states(structure)
```

### MetricsCalculator

예측/실제 값의 메트릭을 계산합니다.

```python
from models.validation.metrics import MetricsCalculator

calculator = MetricsCalculator()

# 오차 계산
mae = calculator.calculate_mae(predicted, actual)
rmse = calculator.calculate_rmse(predicted, actual)
r2 = calculator.calculate_r2(predicted, actual)
```

---

## Data Module (`data/`)

### CacheManager

계산 결과 캐싱을 관리합니다.

```python
from data.cache.manager import CacheManager

cache = CacheManager(
    cache_dir=Path("./cache"),
    memory_max_size=1000
)

# 캐시 저장/조회
cache.set("structure_001", dft_result)
result = cache.get("structure_001")
```

### StorageBackendManager

다양한 저장소 백엔드를 지원합니다.

```python
from data.storage.backend import StorageBackendManager, BackendConfig, StorageBackend

config = BackendConfig(
    backend_type=StorageBackend.LOCAL,
    base_path="./data"
)
storage = StorageBackendManager(config)

# 파일 저장/조회
await storage.store_file("results/calc_001.json", data)
result = await storage.retrieve_file("results/calc_001.json")
```

**Supported Backends:**
- `LOCAL`: 로컬 파일시스템
- `S3`: AWS S3
- `AZURE`: Azure Blob Storage
- `GCS`: Google Cloud Storage
- `FTP`: FTP 서버

---

## Configuration (`config/`)

### Settings

```python
from config.settings import load_config, SystemSettings

# YAML 파일에서 로드
config = load_config("config.yaml")

# 직접 생성
config = SystemSettings(
    dft_code="vasp",
    ml_model_path="./models/predictor.pt",
    cache_dir="./cache",
    max_iterations=100
)
```

### Configuration File Example

```yaml
# config.yaml
system:
  dft_code: vasp
  max_iterations: 100
  convergence_threshold: 0.01

ml:
  model_path: ./models/predictor.pt
  hidden_dim: 64
  n_layers: 3

dft:
  encut: 500
  kpoints: [4, 4, 4]
  xc_functional: PBE

optimization:
  population_size: 20
  mutation_rate: 0.3
  crossover_rate: 0.5
```

---

## Utility Functions (`utils/`)

### helpers.py

```python
from utils.helpers import (
    generate_unique_id,
    calculate_distances,
    calculate_volume,
    convert_to_cartesian,
    load_json_file,
    format_time
)

# 고유 ID 생성
id = generate_unique_id(prefix="calc")  # calc_20231215_120000_a1b2c3d4

# 원자간 거리 계산
distances = calculate_distances(positions, lattice)

# 셀 부피 계산
volume = calculate_volume(lattice)  # Angstrom^3

# 좌표 변환
cartesian = convert_to_cartesian(fractional, lattice)
```

### constants.py

```python
from utils.constants import (
    ATOMIC_MASS,
    MIN_ATOMIC_DISTANCE,
    MAX_ATOMIC_DISTANCE,
    DEFAULT_CUTOFF
)

mass = ATOMIC_MASS[14]  # Si: 28.085
```

---

## Scripts

### optimize.py

```bash
python scripts/optimize.py \
    --structure initial.cif \
    --target-properties '{"band_gap": 1.5}' \
    --max-iterations 100 \
    --output-dir ./results
```

### train.py

```bash
python scripts/train.py \
    --data-dir ./training_data \
    --model-type gnn \
    --epochs 100 \
    --output-dir ./models
```

### analyze.py

```bash
python scripts/analyze.py \
    ./results \
    --analysis-type all \
    --output-dir ./analysis
```

---

## Error Handling

```python
from models.dft.parser import OutputParserError

try:
    results = parser.parse_output(calc_dir)
except OutputParserError as e:
    print(f"Parsing failed: {e}")
```

**Common Exceptions:**
- `OutputParserError`: DFT 출력 파싱 실패
- `ValidationError`: 구조 검증 실패
- `ConvergenceError`: 최적화 수렴 실패
- `CalculationError`: DFT 계산 실패

---

## Type Hints

모든 공개 API는 타입 힌트를 제공합니다:

```python
def calculate_distances(
    positions: np.ndarray,
    lattice: np.ndarray
) -> np.ndarray:
    ...

async def run_optimization(
    initial_structure: Structure,
    target_properties: Dict[str, float],
    max_iterations: int = 100
) -> OptimizationResult:
    ...
```
