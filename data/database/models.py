from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime,
    ForeignKey, JSON, Enum, Table, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class CalculationStatus(enum.Enum):
    """계산 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class DataSource(enum.Enum):
    """데이터 출처"""
    DFT = "dft"
    ML = "ml"
    EXPERIMENTAL = "experimental"
    LITERATURE = "literature"


class Structure(Base):
    """구조 정보 모델"""
    __tablename__ = "structures"

    id = Column(String(36), primary_key=True)
    formula = Column(String(100), index=True)
    atomic_numbers = Column(JSON)  # 원자 번호 배열
    positions = Column(JSON)  # 원자 위치 [분율 좌표]
    lattice_vectors = Column(JSON)  # 격자 벡터 [3x3 행렬]
    space_group = Column(String(20))  # 공간군
    created_at = Column(DateTime, default=datetime.utcnow)
    modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    source = Column(Enum(DataSource))
    metadata = Column(JSON)  # 추가 메타데이터

    # 관계
    calculations = relationship("Calculation", back_populates="structure")
    mutations = relationship("Mutation", back_populates="structure")


class Calculation(Base):
    """DFT 계산 결과 모델"""
    __tablename__ = "calculations"

    id = Column(String(36), primary_key=True)
    structure_id = Column(String(36), ForeignKey("structures.id"))
    status = Column(Enum(CalculationStatus))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # 에너지 관련
    total_energy = Column(Float)
    energy_per_atom = Column(Float)
    formation_energy = Column(Float)

    # 힘과 응력
    forces = Column(JSON)  # 원자에 작용하는 힘
    stress_tensor = Column(JSON)  # 응력 텐서

    # 전자 구조
    band_gap = Column(Float)
    magnetic_moment = Column(Float)
    dos = Column(JSON)  # 상태 밀도

    # 기타 정보
    convergence = Column(Boolean)
    calculation_parameters = Column(JSON)
    error_messages = Column(JSON)

    # 관계
    structure = relationship("Structure", back_populates="calculations")


class Mutation(Base):
    """Mutation 결과 모델"""
    __tablename__ = "mutations"

    id = Column(String(36), primary_key=True)
    structure_id = Column(String(36), ForeignKey("structures.id"))
    parent_id = Column(String(36), ForeignKey("mutations.id"), nullable=True)
    mutation_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    # 평가 점수
    stability_score = Column(Float)
    validity_score = Column(Float)
    fitness_score = Column(Float)

    # 변화 정보
    changes = Column(JSON)  # 구조 변화 설명
    parameters = Column(JSON)  # Mutation 파라미터

    # 관계
    structure = relationship("Structure", back_populates="mutations")
    children = relationship("Mutation",
                            backref=ForeignKey("mutations.parent_id"))


class OptimizationPath(Base):
    """최적화 경로 모델"""
    __tablename__ = "optimization_paths"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    target_properties = Column(JSON)  # 목표 물성
    accepted_error = Column(JSON)  # 허용 오차

    # 경로 정보
    steps = relationship("PathStep", back_populates="path")
    success = Column(Boolean)
    convergence_achieved = Column(Boolean)
    total_iterations = Column(Integer)
    computation_time = Column(Float)


class PathStep(Base):
    """경로 단계 모델"""
    __tablename__ = "path_steps"

    id = Column(String(36), primary_key=True)
    path_id = Column(String(36), ForeignKey("optimization_paths.id"))
    step_number = Column(Integer)
    step_type = Column(String(50))

    # 구조 정보
    initial_structure_id = Column(String(36), ForeignKey("structures.id"))
    final_structure_id = Column(String(36), ForeignKey("structures.id"))

    # 에너지 정보
    energy_initial = Column(Float)
    energy_final = Column(Float)
    energy_barrier = Column(Float)

    # 평가 정보
    success = Column(Boolean)
    confidence_score = Column(Float)
    evaluation_metrics = Column(JSON)

    # 관계
    path = relationship("OptimizationPath", back_populates="steps")


class MLModel(Base):
    """ML 모델 상태 모델"""
    __tablename__ = "ml_models"

    id = Column(String(36), primary_key=True)
    model_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

    # 모델 정보
    parameters = Column(JSON)  # 모델 파라미터
    architecture = Column(JSON)  # 모델 구조
    training_history = Column(JSON)  # 학습 이력

    # 성능 메트릭스
    performance_metrics = Column(JSON)
    validation_metrics = Column(JSON)