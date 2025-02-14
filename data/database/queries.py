from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, not_, desc
from uuid import UUID
from datetime import datetime

from models import (
    Structure, Calculation, Mutation,
    OptimizationPath, PathStep, MLModel
)
from schemas import (
    StructureCreate, CalculationCreate, MutationCreate,
    OptimizationPathCreate, PathStepCreate, MLModelCreate
)


class DatabaseQueries:
    """데이터베이스 쿼리 클래스"""

    @staticmethod
    async def create_structure(db: Session, structure: StructureCreate) -> Structure:
        """구조 생성"""
        db_structure = Structure(**structure.dict())
        db.add(db_structure)
        db.commit()
        db.refresh(db_structure)
        return db_structure

    @staticmethod
    async def get_structure(db: Session, structure_id: UUID) -> Optional[Structure]:
        """구조 조회"""
        return db.query(Structure).filter(Structure.id == structure_id).first()

    @staticmethod
    async def get_structures(
            db: Session,
            skip: int = 0,
            limit: int = 100,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Structure]:
        """구조 목록 조회"""
        query = db.query(Structure)

        if filters:
            for key, value in filters.items():
                if hasattr(Structure, key):
                    query = query.filter(getattr(Structure, key) == value)

        return query.offset(skip).limit(limit).all()

    @staticmethod
    async def create_calculation(db: Session, calc: CalculationCreate) -> Calculation:
        """계산 생성"""
        db_calc = Calculation(**calc.dict())
        db.add(db_calc)
        db.commit()
        db.refresh(db_calc)
        return db_calc

    @staticmethod
    async def update_calculation_status(
            db: Session,
            calc_id: UUID,
            status: str,
            results: Optional[Dict] = None
    ) -> Calculation:
        """계산 상태 업데이트"""
        db_calc = db.query(Calculation).filter(Calculation.id == calc_id).first()
        if db_calc:
            db_calc.status = status
            if status == "completed" and results:
                for key, value in results.items():
                    if hasattr(db_calc, key):
                        setattr(db_calc, key, value)
            db.commit()
            db.refresh(db_calc)
        return db_calc

    @staticmethod
    async def get_calculations_by_structure(
            db: Session,
            structure_id: UUID
    ) -> List[Calculation]:
        """구조별 계산 결과 조회"""
        return db.query(Calculation) \
            .filter(Calculation.structure_id == structure_id) \
            .order_by(desc(Calculation.created_at)) \
            .all()

    @staticmethod
    async def create_mutation(db: Session, mutation: MutationCreate) -> Mutation:
        """Mutation 생성"""
        db_mutation = Mutation(**mutation.dict())
        db.add(db_mutation)
        db.commit()
        db.refresh(db_mutation)
        return db_mutation

    @staticmethod
    async def get_mutation_history(
            db: Session,
            structure_id: UUID
    ) -> List[Mutation]:
        """Mutation 이력 조회"""
        return db.query(Mutation) \
            .filter(Mutation.structure_id == structure_id) \
            .order_by(Mutation.created_at) \
            .all()

    @staticmethod
    async def create_optimization_path(
            db: Session,
            path: OptimizationPathCreate
    ) -> OptimizationPath:
        """최적화 경로 생성"""
        db_path = OptimizationPath(**path.dict())
        db.add(db_path)
        db.commit()
        db.refresh(db_path)
        return db_path

    @staticmethod
    async def add_path_step(
            db: Session,
            step: PathStepCreate
    ) -> PathStep:
        """경로 단계 추가"""
        db_step = PathStep(**step.dict())
        db.add(db_step)
        db.commit()
        db.refresh(db_step)
        return db_step

    @staticmethod
    async def get_optimization_path(
            db: Session,
            path_id: UUID
    ) -> Optional[OptimizationPath]:
        """최적화 경로 조회"""
        return db.query(OptimizationPath) \
            .filter(OptimizationPath.id == path_id) \
            .first()

    @staticmethod
    async def get_similar_structures(
            db: Session,
            structure: Structure,
            threshold: float = 0.9,
            limit: int = 10
    ) -> List[Structure]:
        """유사 구조 검색"""
        # 구조 유사도 계산 로직 구현 필요
        pass

    @staticmethod
    async def get_property_range(
            db: Session,
            property_name: str
    ) -> Tuple[float, float]:
        """물성값 범위 조회"""
        result = db.query(
            func.min(getattr(Calculation, property_name)),
            func.max(getattr(Calculation, property_name))
        ).first()
        return result if result else (None, None)

    @staticmethod
    async def search_structures(
            db: Session,
            query: Dict[str, Any],
            limit: int = 10
    ) -> List[Structure]:
        """구조 검색"""
        base_query = db.query(Structure)

        # 검색 조건 적용
        if 'formula' in query:
            base_query = base_query.filter(
                Structure.formula.like(f"%{query['formula']}%")
            )

        if 'property_range' in query:
            for prop, (min_val, max_val) in query['property_range'].items():
                base_query = base_query.join(Calculation) \
                    .filter(and_(
                    getattr(Calculation, prop) >= min_val,
                    getattr(Calculation, prop) <= max_val
                ))

        return base_query.limit(limit).all()

    @staticmethod
    async def get_calculation_statistics(
            db: Session,
            property_name: str
    ) -> Dict[str, float]:
        """계산 통계 조회"""
        result = db.query(
            func.avg(getattr(Calculation, property_name)).label('mean'),
            func.stddev(getattr(Calculation, property_name)).label('std'),
            func.count(getattr(Calculation, property_name)).label('count')
        ).first()

        return {
            'mean': result.mean,
            'std': result.std,
            'count': result.count
        } if result else None