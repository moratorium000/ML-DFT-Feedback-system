from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from interfaces import Structure, ValidationResult
from protocols import IStructureValidator
from utils.logger import get_logger


@dataclass
class ValidationParameters:
    """검증 파라미터"""
    min_atomic_distance: float = 0.7  # Å
    max_atomic_distance: float = 3.0  # Å
    min_cell_angle: float = 30.0  # degrees
    max_cell_angle: float = 150.0  # degrees
    max_volume_change: float = 0.3  # 30%
    coordination_tolerance: float = 0.2  # 배위수 허용 오차
    bond_angle_tolerance: float = 15.0  # degrees


class StructureValidator(IStructureValidator):
    """구조 검증기"""

    def __init__(self, params: Optional[ValidationParameters] = None):
        self.params = params or ValidationParameters()
        self.logger = get_logger(__name__)

    async def validate(self, structure: Structure) -> ValidationResult:
        """구조 유효성 검증"""
        try:
            # 물리적 제약조건 검사
            physical_checks = await self.check_physical_constraints(structure)

            # 안정성 분석
            stability_analysis = await self.analyze_stability(structure)

            # 대칭성 검증
            symmetry_check = await self.verify_symmetry(structure)

            # 종합 평가
            is_valid = all(physical_checks.values())
            stability_score = np.mean(list(stability_analysis.values()))

            return ValidationResult(
                is_valid=is_valid,
                stability_score=stability_score,
                validation_details={
                    'physical_checks': physical_checks,
                    'stability_analysis': stability_analysis,
                    'symmetry_check': symmetry_check
                },
                error_messages=self._generate_error_messages(
                    physical_checks,
                    stability_analysis
                )
            )

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                stability_score=0.0,
                validation_details={},
                error_messages=[str(e)]
            )

    async def check_physical_constraints(self, structure: Structure) -> Dict[str, bool]:
        """물리적 제약조건 검사"""
        # 원자간 거리 검사
        distances_ok = self._check_atomic_distances(structure)

        # 격자 각도 검사
        angles_ok = self._check_cell_angles(structure)

        # 부피 검사
        volume_ok = self._check_cell_volume(structure)

        # 주기성 검사
        periodicity_ok = self._check_periodicity(structure)

        return {
            'atomic_distances': distances_ok,
            'cell_angles': angles_ok,
            'cell_volume': volume_ok,
            'periodicity': periodicity_ok
        }

    async def analyze_stability(self, structure: Structure) -> Dict[str, float]:
        """안정성 분석"""
        return {
            'coordination_score': self._analyze_coordination(structure),
            'bond_angles_score': self._analyze_bond_angles(structure),
            'density_score': self._analyze_density(structure),
            'packing_score': self._analyze_packing(structure)
        }

    async def verify_symmetry(self, structure: Structure) -> Dict[str, any]:
        """대칭성 검증"""
        # 공간군 분석
        spacegroup = self._determine_spacegroup(structure)

        # 대칭 연산 확인
        symmetry_ops = self._find_symmetry_operations(structure)

        # 등가 위치 검사
        equivalent_sites = self._check_equivalent_sites(structure)

        return {
            'spacegroup': spacegroup,
            'symmetry_operations': symmetry_ops,
            'equivalent_sites': equivalent_sites
        }

    def _check_atomic_distances(self, structure: Structure) -> bool:
        """원자간 거리 검사"""
        positions = structure.positions
        lattice = structure.lattice_vectors

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                # 최소 이미지 규약 적용
                diff = positions[i] - positions[j]
                diff = diff - np.round(diff)
                cart_diff = np.dot(diff, lattice)
                distance = np.linalg.norm(cart_diff)

                if distance < self.params.min_atomic_distance:
                    return False

        return True

    def _check_cell_angles(self, structure: Structure) -> bool:
        """격자 각도 검사"""
        alpha = structure.cell_params['alpha']
        beta = structure.cell_params['beta']
        gamma = structure.cell_params['gamma']

        return all(
            self.params.min_cell_angle <= angle <= self.params.max_cell_angle
            for angle in [alpha, beta, gamma]
        )

    def _analyze_coordination(self, structure: Structure) -> float:
        """배위 환경 분석"""
        scores = []
        for i, pos in enumerate(structure.positions):
            # 주변 원자 찾기
            neighbors = self._find_neighbors(structure, i)

            # 예상 배위수
            expected_cn = self._get_expected_coordination(
                structure.atomic_numbers[i]
            )

            # 실제 배위수
            actual_cn = len(neighbors)

            # 점수 계산
            deviation = abs(actual_cn - expected_cn)
            score = max(0, 1 - deviation * self.params.coordination_tolerance)
            scores.append(score)

        return np.mean(scores)

    def _analyze_bond_angles(self, structure: Structure) -> float:
        """결합각 분석"""
        scores = []
        for i, pos in enumerate(structure.positions):
            neighbors = self._find_neighbors(structure, i)
            if len(neighbors) < 2:
                continue

            # 모든 결합각 계산
            angles = self._calculate_bond_angles(
                pos,
                [structure.positions[j] for j in neighbors]
            )

            # 이상적인 각도와 비교
            ideal_angles = self._get_ideal_angles(len(neighbors))
            deviations = [min(abs(a - b) for b in ideal_angles)
                          for a in angles]

            # 점수 계산
            score = max(0, 1 - np.mean(deviations) /
                        self.params.bond_angle_tolerance)
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    def _find_neighbors(self,
                        structure: Structure,
                        center_idx: int,
                        max_distance: Optional[float] = None) -> List[int]:
        """주변 원자 찾기"""
        if max_distance is None:
            max_distance = self.params.max_atomic_distance

        neighbors = []
        center_pos = structure.positions[center_idx]

        for i, pos in enumerate(structure.positions):
            if i == center_idx:
                continue

            diff = pos - center_pos
            diff = diff - np.round(diff)  # 최소 이미지 규약
            cart_diff = np.dot(diff, structure.lattice_vectors)
            distance = np.linalg.norm(cart_diff)

            if distance <= max_distance:
                neighbors.append(i)

        return neighbors

    def _generate_error_messages(self,
                                 physical_checks: Dict[str, bool],
                                 stability_analysis: Dict[str, float]) -> List[str]:
        """오류 메시지 생성"""
        messages = []

        # 물리적 제약조건 위반 확인
        for check, passed in physical_checks.items():
            if not passed:
                messages.append(f"Physical constraint violated: {check}")

        # 안정성 문제 확인
        for analysis, score in stability_analysis.items():
            if score < 0.5:
                messages.append(
                    f"Low stability score in {analysis}: {score:.2f}"
                )

        return messages