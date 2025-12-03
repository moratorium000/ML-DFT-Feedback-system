from typing import Dict, List, Optional, Union
from pathlib import Path
import asyncio
import logging
from datetime import datetime

from .interfaces import (
    Structure,
    MutationResult,
    DFTResult,
    PathStep,
    ValidationResult,
    PredictionResult
)

from .protocols import (
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
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.structure_validator = self._init_validator()

    def _init_validator(self) -> IStructureValidator:
        """구조 검증기 초기화"""
        from models.mutation.validator import StructureValidator, ValidationParameters
        params = ValidationParameters(**self.config.validation_settings)
        return StructureValidator(params)

    def _generate_id(self, structure: Structure) -> str:
        """구조 기반 고유 ID 생성"""
        import hashlib
        content = f"{structure.formula}_{structure.atomic_numbers.tobytes().hex()}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"proto_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash_val}"

    async def _save_prototype(self, prototype_id: str, structure: Structure, metadata: Dict):
        """Prototype 저장"""
        import json
        save_path = self.cache_dir / f"{prototype_id}.{self.config.data_format}"
        data = {
            "metadata": metadata,
            "structure": {
                "formula": structure.formula,
                "atomic_numbers": structure.atomic_numbers.tolist(),
                "positions": structure.positions.tolist(),
                "lattice_vectors": structure.lattice_vectors.tolist(),
                "cell_params": structure.cell_params
            }
        }
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Prototype saved: {save_path}")

    async def load_structure(self, path: Union[str, Path]) -> Structure:
        """구조 파일 로드"""
        import json
        path = Path(path)
        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
            return Structure(**data.get('structure', data))
        else:
            from ase.io import read
            import numpy as np
            atoms = read(str(path))
            return Structure(
                atomic_numbers=np.array(atoms.get_atomic_numbers()),
                positions=atoms.get_scaled_positions(),
                lattice_vectors=np.array(atoms.get_cell()),
                cell_params={"a": atoms.cell.lengths()[0], "b": atoms.cell.lengths()[1],
                            "c": atoms.cell.lengths()[2]},
                formula=atoms.get_chemical_formula()
            )

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
        self._active_jobs: Dict[str, Dict] = {}

    def _init_calculator(self):
        """DFT 계산기 초기화"""
        from models.dft.calculator import DFTCalculator
        config_dict = {
            'dft_code': self.config.code,
            'work_dir': str(Path('.').resolve() / 'dft_work'),
            'default_params': self.config.input_parameters,
            'convergence_criteria': self.config.convergence_criteria,
            'polling_interval': 5
        }
        return DFTCalculator(config_dict)

    def _prepare_calculation(self, step: PathStep) -> Dict:
        """계산 입력 준비"""
        return {
            "structure": step.final_structure,
            "parameters": self.config.input_parameters,
            "convergence": self.config.convergence_criteria,
            "parallel": self.config.parallel_settings
        }

    async def _submit_job(self, calc_inputs: Dict) -> str:
        """계산 작업 제출"""
        import uuid
        job_id = f"dft_{uuid.uuid4().hex[:8]}"
        self._active_jobs[job_id] = {
            "inputs": calc_inputs,
            "status": "submitted",
            "start_time": datetime.now()
        }
        await self.job_queue.put((job_id, calc_inputs))
        self.logger.info(f"Job submitted: {job_id}")
        return job_id

    async def _monitor_calculation(self, job_id: str) -> Dict:
        """계산 모니터링"""
        job_info = self._active_jobs.get(job_id)
        if not job_info:
            raise ValueError(f"Job not found: {job_id}")

        # 계산 실행 (실제로는 외부 DFT 코드 호출)
        result = await self.calculator.calculate(
            structure=job_info["inputs"]["structure"],
            parameters=job_info["inputs"]["parameters"]
        )
        job_info["status"] = "completed"
        job_info["result"] = result
        return result

    def _process_result(self, result: Dict) -> DFTResult:
        """결과 처리 및 DFTResult 변환"""
        import numpy as np
        return DFTResult(
            initial_structure=result.get("initial_structure"),
            final_structure=result.get("final_structure"),
            total_energy=result.get("total_energy", 0.0),
            energy_per_atom=result.get("energy_per_atom", 0.0),
            formation_energy=result.get("formation_energy", 0.0),
            forces=np.array(result.get("forces", [])),
            stress_tensor=np.array(result.get("stress", np.zeros((3, 3)))),
            band_gap=result.get("band_gap"),
            dos=result.get("dos"),
            band_structure=result.get("band_structure"),
            convergence=result.get("converged", False),
            calculation_time=result.get("calculation_time", 0.0),
            error_messages=result.get("errors", [])
        )

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

    def _init_property_predictor(self) -> IModelPredictor:
        """물성 예측 모델 초기화"""
        from models.ml.predictor import MLPredictor
        hidden_dim = self.config.model_parameters.get('hidden_layers', [128])[0]
        # MLPredictor는 내부적으로 PropertyPredictor와 PathPredictor를 모두 포함
        return MLPredictor(
            input_dim=4,  # 원자 특성 차원: [원자번호, 전기음성도, 반지름, 이온화에너지]
            hidden_dim=hidden_dim,
            device=self.config.device
        )

    def _init_path_predictor(self) -> IModelPredictor:
        """경로 예측 모델 초기화"""
        from models.ml.predictor import MLPredictor
        hidden_dim = self.config.model_parameters.get('hidden_layers', [128])[0]
        return MLPredictor(
            input_dim=4,
            hidden_dim=hidden_dim,
            device=self.config.device
        )

    def _prepare_training_data(self, dft_results: List[DFTResult]) -> Dict:
        """학습 데이터 준비"""
        import numpy as np
        structures = []
        targets = []

        for result in dft_results:
            if result.final_structure is not None:
                structures.append({
                    "positions": result.final_structure.positions,
                    "atomic_numbers": result.final_structure.atomic_numbers,
                    "lattice": result.final_structure.lattice_vectors
                })
                targets.append({
                    "total_energy": result.total_energy,
                    "band_gap": result.band_gap,
                    "formation_energy": result.formation_energy
                })

        return {
            "structures": structures,
            "targets": targets,
            "batch_size": self.config.training_parameters.get('batch_size', 32)
        }

    async def predict_paths(self,
                            structure: Structure,
                            target_properties: Dict[str, float]) -> List[PathStep]:
        """경로 예측"""
        # 물성 예측
        property_prediction = await self.property_predictor.predict_properties(
            structure
        )

        # 목표 구조 생성 (target_properties 기반)
        # 일단은 현재 구조를 목표로 사용 (실제로는 역설계 로직 필요)
        target_structure = structure

        # 경로 예측
        paths = await self.path_predictor.predict_path(
            structure,
            target_structure
        )

        return paths

    async def update_models(self, dft_results: List[DFTResult]):
        """모델 업데이트"""
        # 학습 데이터 준비
        training_data = self._prepare_training_data(dft_results)

        # TODO: 실제 모델 학습 로직 구현
        # 현재는 로깅만 수행
        self.logger.info(f"Model update requested with {len(training_data.get('structures', []))} samples")


class PathManager:
    """경로 관리"""

    def __init__(self, config: PathConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.mutation_generator = self._init_mutation_generator()
        self.path_optimizer = self._init_path_optimizer()

    def _init_mutation_generator(self) -> IMutationGenerator:
        """변이 생성기 초기화"""
        from models.mutation.generator import MutationGenerator, MutationParameters
        params = MutationParameters()
        return MutationGenerator(params)

    def _init_path_optimizer(self):
        """경로 최적화기 초기화"""
        from models.mutation.optimizer import MutationOptimizer, OptimizerParameters
        params = OptimizerParameters(**self.config.optimization_parameters)
        return MutationOptimizer(params)

    async def _evaluate_mutations(self,
                                  mutations: List[MutationResult],
                                  target_properties: Dict[str, float]) -> List[Dict]:
        """변이 평가"""
        evaluations = []
        for mutation in mutations:
            score = 0.0
            if mutation.success:
                score = mutation.stability_score * 0.5 + mutation.validity_score * 0.5
            evaluations.append({
                "mutation": mutation,
                "score": score,
                "valid": mutation.success
            })
        return evaluations

    def _evaluate_property_match(self,
                                 path: Union[PathStep, List[PathStep]],
                                 target_properties: Dict[str, float]) -> float:
        """물성 일치도 평가"""
        if isinstance(path, list):
            if not path:
                return 0.0
            path = path[-1]  # 마지막 단계 사용

        if path.ml_predictions is None:
            return 0.0

        scores = []
        for prop, target in target_properties.items():
            predicted = path.ml_predictions.get(prop, 0)
            if target != 0:
                error = abs(predicted - target) / abs(target)
                scores.append(max(0.0, 1.0 - error))
            else:
                scores.append(1.0 if predicted == 0 else 0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _evaluate_feasibility(self, path: Union[PathStep, List[PathStep]]) -> float:
        """실현 가능성 평가"""
        if isinstance(path, list):
            if not path:
                return 0.0
            # 모든 단계의 평균 신뢰도
            confidences = [step.confidence for step in path]
            return sum(confidences) / len(confidences)
        return path.confidence

    def _evaluate_efficiency(self, path: Union[PathStep, List[PathStep]]) -> float:
        """효율성 평가 (에너지 장벽 기반)"""
        if isinstance(path, list):
            if not path:
                return 0.0
            barriers = [step.energy_barrier for step in path if step.energy_barrier]
            if not barriers:
                return 0.5
            max_barrier = max(barriers)
            return max(0.0, 1.0 - max_barrier / 1.0)  # 1 eV 기준
        if path.energy_barrier is None:
            return 0.5
        return max(0.0, 1.0 - path.energy_barrier / 1.0)

    async def evaluate_and_select_path(self,
                                       paths: List[PathStep],
                                       target_properties: Dict[str, float]) -> List[PathStep]:
        """경로 평가 및 선택"""
        evaluated_paths = []
        all_mutations = []

        for path in paths:
            # 경로 평가
            evaluation = await self._evaluate_path(path, target_properties)

            # Mutation 생성 및 평가 (final_structure에서 생성)
            if path.final_structure is not None:
                mutations = await self.mutation_generator.generate(path.final_structure)
                mutation_evaluations = await self._evaluate_mutations(
                    mutations,
                    target_properties
                )
                all_mutations.extend(mutations)
            else:
                mutation_evaluations = []

            # 결과 결합
            evaluated_paths.append({
                "path": path,
                "evaluation": evaluation,
                "mutations": mutation_evaluations
            })

        # 최적 경로 선택
        selected_path = self._select_best_path(evaluated_paths)

        # 경로 최적화 (mutations가 있는 경우)
        if all_mutations:
            optimized_mutations = await self.path_optimizer.optimize(
                all_mutations,
                target_properties
            )
            # MutationResult를 PathStep으로 변환
            for mutation in optimized_mutations[:len(selected_path)]:
                if mutation.success:
                    # 기존 경로에 최적화된 구조 적용
                    pass

        return selected_path

    async def _evaluate_path(self,
                             path: Union[PathStep, List[PathStep]],
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
        if isinstance(best_path, list):
            return best_path
        return [best_path]