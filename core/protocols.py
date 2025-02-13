from typing import Protocol, List, Dict, Optional
from interfaces import (
    Structure,
    MutationResult,
    ValidationResult,
    PredictionResult,
    PathStep,
    DFTResult
)


class IStructureValidator(Protocol):
    """구조 검증 프로토콜"""

    async def validate(self, structure: Structure) -> ValidationResult:
        """구조 유효성 검사"""
        ...

    async def check_physical_constraints(self, structure: Structure) -> Dict[str, bool]:
        """물리적 제약조건 검사"""
        ...

    async def analyze_stability(self, structure: Structure) -> Dict[str, float]:
        """안정성 분석"""
        ...

    async def verify_symmetry(self, structure: Structure) -> Dict[str, any]:
        """대칭성 검증"""
        ...


class IMutationGenerator(Protocol):
    """Mutation 생성 프로토콜"""

    async def generate(self,
                       structure: Structure,
                       mutation_params: Dict[str, any]) -> List[MutationResult]:
        """Mutation 생성"""
        ...

    async def validate_mutation(self,
                                original: Structure,
                                mutated: Structure) -> ValidationResult:
        """Mutation 유효성 검사"""
        ...

    async def estimate_quality(self, mutation: MutationResult) -> float:
        """Mutation 품질 추정"""
        ...

    async def get_mutation_probability(self,
                                       structure: Structure,
                                       mutation_type: str) -> float:
        """Mutation 확률 계산"""
        ...


class IModelPredictor(Protocol):
    """ML 모델 예측 프로토콜"""

    async def predict_properties(self, structure: Structure) -> PredictionResult:
        """물성 예측"""
        ...

    async def predict_path(self,
                           initial: Structure,
                           target: Structure) -> List[PathStep]:
        """경로 예측"""
        ...

    async def estimate_uncertainty(self,
                                   prediction: PredictionResult) -> Dict[str, float]:
        """불확실성 추정"""
        ...

    async def validate_prediction(self,
                                  prediction: PredictionResult,
                                  actual: DFTResult) -> Dict[str, float]:
        """예측 검증"""
        ...


class IPathOptimizer(Protocol):
    """경로 최적화 프로토콜"""

    async def optimize_path(self,
                            path: List[PathStep],
                            constraints: Dict[str, any]) -> List[PathStep]:
        """경로 최적화"""
        ...

    async def evaluate_path(self,
                            path: List[PathStep],
                            target_properties: Dict[str, float]) -> Dict[str, float]:
        """경로 평가"""
        ...

    async def find_transition_states(self,
                                     path: List[PathStep]) -> List[Structure]:
        """전이 상태 탐색"""
        ...


class IDFTCalculator(Protocol):
    """DFT 계산 프로토콜"""

    async def calculate(self,
                        structure: Structure,
                        calc_params: Dict[str, any]) -> DFTResult:
        """DFT 계산 수행"""
        ...

    async def check_convergence(self,
                                calc_result: DFTResult) -> Dict[str, bool]:
        """수렴성 검사"""
        ...

    async def estimate_cost(self,
                            structure: Structure,
                            calc_params: Dict[str, any]) -> Dict[str, float]:
        """계산 비용 추정"""
        ...


class IDataManager(Protocol):
    """데이터 관리 프로토콜"""

    async def store_structure(self, structure: Structure) -> str:
        """구조 저장"""
        ...

    async def store_result(self, result: DFTResult) -> str:
        """결과 저장"""
        ...

    async def query_similar_structures(self,
                                       structure: Structure,
                                       threshold: float) -> List[Structure]:
        """유사 구조 검색"""
        ...

    async def get_calculation_history(self,
                                      structure_id: str) -> List[DFTResult]:
        """계산 이력 조회"""
        ...


class IFeedbackAnalyzer(Protocol):
    """피드백 분석 프로토콜"""

    async def analyze_feedback(self,
                               prediction: PredictionResult,
                               actual: DFTResult) -> Dict[str, any]:
        """피드백 분석"""
        ...

    async def suggest_improvements(self,
                                   analysis: Dict[str, any]) -> Dict[str, any]:
        """개선사항 제안"""
        ...

    async def update_strategy(self,
                              feedback: Dict[str, any]) -> Dict[str, any]:
        """전략 업데이트"""
        ...