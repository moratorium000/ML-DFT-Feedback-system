from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
from enum import Enum

class CalculationStatus(Enum):
    """계산 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class Structure:
    """원자 구조"""
    atomic_numbers: np.ndarray      # 원자 번호 배열
    positions: np.ndarray          # 원자 위치 (분율 좌표)
    lattice_vectors: np.ndarray    # 격자 벡터
    cell_params: Dict[str, float]  # 격자 상수 및 각도
    formula: str                   # 화학식
    space_group: Optional[str] = None  # 공간군
    charge: float = 0.0            # 전체 전하량
    spin: float = 0.0             # 스핀 상태
    constraints: Optional[Dict] = None  # 구조 제약 조건

@dataclass
class MutationResult:
    """Mutation 결과"""
    original_structure: Structure   # 원본 구조
    mutated_structure: Structure   # 변형된 구조
    mutation_type: str            # mutation 유형
    changes: Dict[str, any]       # 적용된 변경 사항
    success: bool                # mutation 성공 여부
    stability_score: float       # 안정성 점수
    validity_score: float        # 물리적 타당성 점수
    energy_estimate: float      # 예상 에너지
    generation: int             # mutation 세대
    parent_id: Optional[str]    # 부모 구조 ID
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DFTResult:
    """DFT 계산 결과"""
    initial_structure: Structure   # 초기 구조
    final_structure: Structure     # 최종 구조
    total_energy: float           # 전체 에너지
    energy_per_atom: float        # 원자당 에너지
    formation_energy: float       # 형성 에너지
    forces: np.ndarray            # 원자에 작용하는 힘
    stress_tensor: np.ndarray     # 응력 텐서
    band_gap: Optional[float]     # 밴드갭
    dos: Optional[Dict]           # 상태 밀도
    band_structure: Optional[Dict] # 밴드 구조
    convergence: bool             # 수렴 여부
    calculation_time: float       # 계산 시간
    error_messages: List[str]     # 오류 메시지
    magnetic_moment: Optional[float] = None  # 자기모멘트
    charge_density: Optional[np.ndarray] = None  # 전하 밀도
    additional_properties: Dict[str, any] = field(default_factory=dict)

@dataclass
class PathStep:
    """경로 단계"""
    step_type: str                # 단계 유형
    initial_structure: Structure   # 시작 구조
    final_structure: Structure     # 최종 구조
    energy_initial: float         # 시작 에너지
    energy_final: float           # 최종 에너지
    energy_barrier: Optional[float] = None  # 에너지 장벽
    transformation_matrix: Optional[np.ndarray] = None  # 구조 변환 행렬
    atomic_mapping: Optional[Dict[int, int]] = None  # 원자 매핑
    dft_results: Optional[DFTResult] = None  # DFT 계산 결과
    ml_predictions: Optional[Dict] = None  # ML 예측 결과
    success: bool = False         # 단계 성공 여부
    reversible: bool = False     # 가역성 여부
    confidence: float = 0.0      # 신뢰도 점수

@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool               # 유효성 여부
    stability_score: float       # 안정성 점수
    validation_details: Dict[str, any]  # 검증 세부사항
    error_messages: List[str]    # 오류 메시지

@dataclass
class PredictionResult:
    """예측 결과"""
    predicted_values: Dict[str, float]  # 예측값
    uncertainty: Dict[str, float]      # 불확실성
    confidence_score: float           # 신뢰도 점수
    prediction_details: Dict[str, any] # 예측 세부사항

@dataclass
class OptimizationResult:
    """최적화 결과"""
    final_structure: Structure    # 최종 구조
    property_matches: Dict[str, float]  # 물성 일치도
    optimization_path: List[Dict]  # 최적화 경로
    convergence_achieved: bool    # 수렴 여부
    total_iterations: int         # 총 반복 횟수
    computation_time: float       # 계산 시간
    final_score: float           # 최종 점수
    history: List[Dict] = field(default_factory=list)  # 최적화 히스토리

@dataclass
class MLModelState:
    """ML 모델 상태"""
    model_type: str              # 모델 유형
    training_iterations: int     # 학습 반복 횟수
    performance_metrics: Dict[str, float]  # 성능 지표
    last_update: datetime       # 마지막 업데이트 시간
    model_parameters: Dict[str, any]  # 모델 파라미터
    training_history: List[Dict] = field(default_factory=list)  # 학습 히스토리

@dataclass
class FeedbackResult:
    """피드백 결과"""
    iteration: int              # 반복 횟수
    structure_changes: Dict[str, float]  # 구조 변화
    property_improvements: Dict[str, float]  # 물성 개선
    ml_model_updates: Dict[str, float]  # ML 모델 업데이트
    convergence_metrics: Dict[str, float]  # 수렴성 지표
    timestamp: datetime = field(default_factory=datetime.now)