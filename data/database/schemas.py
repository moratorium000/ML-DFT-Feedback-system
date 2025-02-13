from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from uuid import UUID
from enum import Enum

class DataSource(str, Enum):
    """데이터 출처"""
    DFT = "dft"
    ML = "ml"
    EXPERIMENTAL = "experimental"
    LITERATURE = "literature"

class CalculationStatus(str, Enum):
    """계산 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class StructureBase(BaseModel):
    """구조 기본 스키마"""
    formula: str
    atomic_numbers: List[int]
    positions: List[List[float]]
    lattice_vectors: List[List[float]]
    space_group: Optional[str] = None
    source: DataSource
    metadata: Optional[Dict] = None

class StructureCreate(StructureBase):
    """구조 생성 스키마"""
    pass

class StructureUpdate(BaseModel):
    """구조 업데이트 스키마"""
    formula: Optional[str] = None
    positions: Optional[List[List[float]]] = None
    lattice_vectors: Optional[List[List[float]]] = None
    metadata: Optional[Dict] = None

class Structure(StructureBase):
    """구조 응답 스키마"""
    id: UUID
    created_at: datetime
    modified_at: datetime

    class Config:
        orm_mode = True

class CalculationBase(BaseModel):
    """계산 기본 스키마"""
    structure_id: UUID
    status: CalculationStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_energy: Optional[float] = None
    energy_per_atom: Optional[float] = None
    formation_energy: Optional[float] = None
    forces: Optional[List[List[float]]] = None
    stress_tensor: Optional[List[List[float]]] = None
    band_gap: Optional[float] = None
    magnetic_moment: Optional[float] = None
    dos: Optional[Dict] = None
    convergence: Optional[bool] = None
    calculation_parameters: Dict
    error_messages: Optional[List[str]] = None

class CalculationCreate(CalculationBase):
    """계산 생성 스키마"""
    pass

class CalculationUpdate(BaseModel):
    """계산 업데이트 스키마"""
    status: Optional[CalculationStatus] = None
    completed_at: Optional[datetime] = None
    total_energy: Optional[float] = None
    energy_per_atom: Optional[float] = None
    forces: Optional[List[List[float]]] = None
    error_messages: Optional[List[str]] = None

class Calculation(CalculationBase):
    """계산 응답 스키마"""
    id: UUID

    class Config:
        orm_mode = True

class MutationBase(BaseModel):
    """Mutation 기본 스키마"""
    structure_id: UUID
    parent_id: Optional[UUID] = None
    mutation_type: str
    stability_score: float
    validity_score: float
    fitness_score: float
    changes: Dict
    parameters: Dict

class MutationCreate(MutationBase):
    """Mutation 생성 스키마"""
    pass

class Mutation(MutationBase):
    """Mutation 응답 스키마"""
    id: UUID
    created_at: datetime

    class Config:
        orm_mode = True

class OptimizationPathBase(BaseModel):
    """최적화 경로 기본 스키마"""
    target_properties: Dict[str, float]
    accepted_error: Dict[str, float]
    success: Optional[bool] = None
    convergence_achieved: Optional[bool] = None
    total_iterations: Optional[int] = None
    computation_time: Optional[float] = None

class OptimizationPathCreate(OptimizationPathBase):
    """최적화 경로 생성 스키마"""
    pass

class OptimizationPath(OptimizationPathBase):
    """최적화 경로 응답 스키마"""
    id: UUID
    created_at: datetime
    steps: List['PathStep']

    class Config:
        orm_mode = True

class PathStepBase(BaseModel):
    """경로 단계 기본 스키마"""
    path_id: UUID
    step_number: int
    step_type: str
    initial_structure_id: UUID
    final_structure_id: UUID
    energy_initial: float
    energy_final: float
    energy_barrier: Optional[float] = None
    success: bool
    confidence_score: float
    evaluation_metrics: Dict

class PathStepCreate(PathStepBase):
    """경로 단계 생성 스키마"""
    pass

class PathStep(PathStepBase):
    """경로 단계 응답 스키마"""
    id: UUID

    class Config:
        orm_mode = True

class MLModelBase(BaseModel):
    """ML 모델 기본 스키마"""
    model_type: str
    parameters: Dict
    architecture: Dict
    training_history: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None
    validation_metrics: Optional[Dict] = None

class MLModelCreate(MLModelBase):
    """ML 모델 생성 스키마"""
    pass

class MLModelUpdate(BaseModel):
    """ML 모델 업데이트 스키마"""
    parameters: Optional[Dict] = None
    training_history: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None
    validation_metrics: Optional[Dict] = None

class MLModel(MLModelBase):
    """ML 모델 응답 스키마"""
    id: UUID
    created_at: datetime
    last_updated: datetime

    class Config:
        orm_mode = True

# 순환 참조 해결
OptimizationPath.update_forward_refs()