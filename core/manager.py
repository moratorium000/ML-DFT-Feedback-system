from typing import Dict, List, Optional, Union
from pathlib import Path
import asyncio
import logging
from datetime import datetime

from interfaces import (
    Structure,
    MutationResult,
    DFTResult,
    PathStep,
    ValidationResult,
    PredictionResult
)

from protocols import (
    IStructureValidator,
    IMutationGenerator,
    IModelPredictor
)

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class PrototypeConfig:
    """Prototype 설정"""
    cache_dir: Path = Path(".cache/prototypes")
    validation_settings: Dict = field(default_factory=lambda: {
        'min_atomic_distance': 0.7,  # Å
        'max_atomic_distance': 3.0,  # Å
        'min_cell_angle': 30.0,     # degrees
        'max_cell_angle': 150.0,    # degrees
        'max_volume_change': 0.3    # 30%
    })
    data_format: str = "json"
    backup_enabled: bool = True
    max_cache_size: int = 1000
    cache_ttl: int = 7200  # seconds

@dataclass
class DFTConfig:
    """DFT 계산 설정"""
    code: str = "vasp"
    input_parameters: Dict = field(default_factory=lambda: {
        'xc_functional': 'PBE',
        'energy_cutoff': 520,    # eV
        'kpoints': [2, 2, 2],
        'smearing': 'gaussian',
        'sigma': 0.05,          # eV
        'mixing_beta': 0.7,
        'max_iterations': 100
    })
    convergence_criteria: Dict = field(default_factory=lambda: {
        'energy': 1e-5,    # eV
        'force': 0.02,     # eV/Å
        'stress': 0.1,     # GPa
        'density': 1e-6    # e/Å³
    })
    parallel_settings: Dict = field(default_factory=lambda: {
        'ncore': 4,
        'kpar': 2,
        'lplane': True,
        'npar': 4
    })
    max_time: int = 3600  # seconds
    checkpoint_interval: int = 300  # seconds

@dataclass
class MLConfig:
    """ML 모델 설정"""
    model_type: str = "graph"
    model_parameters: Dict = field(default_factory=lambda: {
        'hidden_layers': [256, 128, 64],
        'activation': 'relu',
        'dropout_rate': 0.1,
        'batch_norm': True
    })
    training_parameters: Dict = field(default_factory=lambda: {
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2
    })
    early_stopping: Dict = field(default_factory=lambda: {
        'patience': 10,
        'min_delta': 0.001
    })
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "cuda"  # or "cpu"

@dataclass
class PathConfig:
    """경로 최적화 설정"""
    optimization_parameters: Dict = field(default_factory=lambda: {
        'max_iterations': 100,
        'convergence_threshold': 0.01,
        'step_size': 0.1,
        'momentum': 0.9
    })
    mutation_settings: Dict = field(default_factory=lambda: {
        'mutation_rate': 0.3,
        'crossover_rate': 0.7,
        'population_size': 50,
        'tournament_size': 3
    })
    diversity_control: Dict = field(default_factory=lambda: {
        'min_diversity': 0.5,
        'diversity_weight': 0.3,
        'novelty_threshold': 0.2
    })
    validation_settings: Dict = field(default_factory=lambda: {
        'property_tolerance': 0.1,
        'structure_tolerance': 0.5,
        'energy_tolerance': 0.05
    })
    dft_validation_interval: int = 5  # iterations
    cache_enabled: bool = True
    history_size: int = 1000

class PrototypeManager:
    """Prototype 관리"""

    def __init__(self, config: PrototypeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(config.cache_dir)
        self.structure_validator = self._init_validator()

    async def register_prototype(self,
                                 structure: Structure,
                                 metadata: Optional[Dict] = None) -> str:
        """Prototype 등록"""
        # 구조 검증
        validation_result = await self.structure_validator.validate(structure)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid structure: {validation_result.error_messages}")

        # Prototype ID 생성
        prototype_id = self._generate_id(structure)

        # 메타데이터 준비
        full_metadata = {
            "id": prototype_id,
            "created_at": datetime.now().isoformat(),
            "source": "user_input",
            **(metadata or {})
        }

        # 저장
        await self._save_prototype(prototype_id, structure, full_metadata)

        return prototype_id


class DFTManager:
    """DFT 계산 관리"""

    def __init__(self, config: DFTConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.calculator = self._init_calculator()
        self.job_queue = asyncio.Queue()

    async def validate_path(self, path: List[PathStep]) -> List[DFTResult]:
        """경로 검증을 위한 DFT 계산"""
        results = []

        for step in path:
            # 계산 준비
            calc_inputs = self._prepare_calculation(step)

            # 작업 제출
            job_id = await self._submit_job(calc_inputs)

            # 계산 모니터링
            result = await self._monitor_calculation(job_id)

            # 결과 처리
            processed_result = self._process_result(result)
            results.append(processed_result)

        return results


class MLManager:
    """ML 모델 관리"""

    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.property_predictor = self._init_property_predictor()
        self.path_predictor = self._init_path_predictor()

    async def predict_paths(self,
                            structure: Structure,
                            target_properties: Dict[str, float]) -> List[PathStep]:
        """경로 예측"""
        # 물성 예측
        property_prediction = await self.property_predictor.predict_properties(
            structure
        )

        # 경로 예측
        paths = await self.path_predictor.predict_paths(
            structure,
            target_properties,
            property_prediction
        )

        return paths

    async def update_models(self, dft_results: List[DFTResult]):
        """모델 업데이트"""
        # 학습 데이터 준비
        training_data = self._prepare_training_data(dft_results)

        # 모델 업데이트
        await self.property_predictor.update(training_data)
        await self.path_predictor.update(training_data)


class PathManager:
    """경로 관리"""

    def __init__(self, config: PathConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.mutation_generator = self._init_mutation_generator()
        self.path_optimizer = self._init_path_optimizer()

    async def evaluate_and_select_path(self,
                                       paths: List[PathStep],
                                       target_properties: Dict[str, float]) -> List[PathStep]:
        """경로 평가 및 선택"""
        evaluated_paths = []

        for path in paths:
            # 경로 평가
            evaluation = await self._evaluate_path(path, target_properties)

            # Mutation 생성 및 평가
            mutations = await self.mutation_generator.generate(path)
            mutation_evaluations = await self._evaluate_mutations(
                mutations,
                target_properties
            )

            # 결과 결합
            evaluated_paths.append({
                "path": path,
                "evaluation": evaluation,
                "mutations": mutation_evaluations
            })

        # 최적 경로 선택
        selected_path = self._select_best_path(evaluated_paths)

        # 경로 최적화
        optimized_path = await self.path_optimizer.optimize(selected_path)

        return optimized_path

    async def _evaluate_path(self,
                             path: List[PathStep],
                             target_properties: Dict[str, float]) -> Dict:
        """경로 평가"""
        return {
            "property_score": self._evaluate_property_match(
                path,
                target_properties
            ),
            "feasibility_score": self._evaluate_feasibility(path),
            "efficiency_score": self._evaluate_efficiency(path)
        }

    def _select_best_path(self, evaluated_paths: List[Dict]) -> List[PathStep]:
        """최적 경로 선택"""
        # 경로 점수 계산
        path_scores = []
        for eval_path in evaluated_paths:
            score = (
                    eval_path["evaluation"]["property_score"] * 0.4 +
                    eval_path["evaluation"]["feasibility_score"] * 0.3 +
                    eval_path["evaluation"]["efficiency_score"] * 0.3
            )
            path_scores.append((eval_path["path"], score))

        # 최고 점수 경로 선택
        best_path, _ = max(path_scores, key=lambda x: x[1])
        return best_path