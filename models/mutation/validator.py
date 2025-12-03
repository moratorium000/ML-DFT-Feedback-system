from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import itertools

from core.interfaces import Structure, ValidationResult
from core.protocols import IStructureValidator
from utils.logger import get_logger
from utils.constants import ELEMENT_SYMBOLS, ATOMIC_RADIUS


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

    async def verify_symmetry(self, structure: Structure) -> Dict[str, Any]:
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

    def _check_cell_volume(self, structure: Structure) -> bool:
        """셀 부피 검사"""
        lattice = np.array(structure.lattice_vectors)
        volume = abs(np.linalg.det(lattice))
        n_atoms = len(structure.atomic_numbers)

        # 원자당 부피 검사 (5-100 Å³/atom 범위)
        volume_per_atom = volume / n_atoms
        return 5.0 < volume_per_atom < 100.0

    def _check_periodicity(self, structure: Structure) -> bool:
        """주기성 검사"""
        # 격자 벡터가 유효한지 확인
        lattice = np.array(structure.lattice_vectors)

        # 격자 벡터가 선형 독립인지 확인
        det = np.linalg.det(lattice)
        if abs(det) < 1e-10:
            return False

        # 모든 위치가 [0, 1) 범위 내에 있는지 확인
        positions = np.array(structure.positions)
        if np.any(positions < -0.5) or np.any(positions > 1.5):
            return False

        return True

    def _get_expected_coordination(self, atomic_number: int) -> int:
        """예상 배위수 반환"""
        # 원소별 일반적인 배위수
        coordination_numbers = {
            1: 1,    # H
            6: 4,    # C
            7: 3,    # N
            8: 2,    # O
            9: 1,    # F
            11: 6,   # Na
            12: 6,   # Mg
            13: 6,   # Al
            14: 4,   # Si
            15: 3,   # P
            16: 2,   # S
            17: 1,   # Cl
            19: 8,   # K
            20: 8,   # Ca
            22: 6,   # Ti
            26: 6,   # Fe
            28: 6,   # Ni
            29: 4,   # Cu
            30: 4,   # Zn
        }
        return coordination_numbers.get(int(atomic_number), 6)

    def _calculate_bond_angles(self,
                               center_pos: np.ndarray,
                               neighbor_positions: List[np.ndarray]) -> List[float]:
        """결합각 계산"""
        angles = []
        neighbor_positions = [np.array(p) for p in neighbor_positions]

        for i, pos1 in enumerate(neighbor_positions):
            for pos2 in neighbor_positions[i + 1:]:
                # 중심에서 이웃으로의 벡터
                v1 = pos1 - center_pos
                v2 = pos2 - center_pos

                # 각도 계산
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)

        return angles

    def _get_ideal_angles(self, coordination_number: int) -> List[float]:
        """배위수에 따른 이상적인 결합각"""
        ideal_angles = {
            2: [180.0],                    # 선형
            3: [120.0],                    # 삼각형 평면
            4: [109.5],                    # 정사면체
            5: [90.0, 120.0],              # 삼각쌍피라미드
            6: [90.0],                     # 정팔면체
            8: [70.5, 109.5],              # 정육면체
        }
        return ideal_angles.get(coordination_number, [90.0])

    def _determine_spacegroup(self, structure: Structure) -> str:
        """공간군 결정 (간단한 구현)"""
        # 실제로는 spglib 등을 사용해야 함
        lattice = np.array(structure.lattice_vectors)
        a = np.linalg.norm(lattice[0])
        b = np.linalg.norm(lattice[1])
        c = np.linalg.norm(lattice[2])

        alpha = structure.cell_params.get('alpha', 90.0)
        beta = structure.cell_params.get('beta', 90.0)
        gamma = structure.cell_params.get('gamma', 90.0)

        # 격자 유형 판별
        is_cubic = np.allclose([a, b, c], a, rtol=0.05) and np.allclose([alpha, beta, gamma], 90.0, atol=5.0)
        is_tetragonal = np.allclose([a, b], a, rtol=0.05) and not np.isclose(a, c, rtol=0.05) and np.allclose([alpha, beta, gamma], 90.0, atol=5.0)
        is_orthorhombic = not np.allclose([a, b, c], a, rtol=0.05) and np.allclose([alpha, beta, gamma], 90.0, atol=5.0)

        if is_cubic:
            return "Fm-3m"  # 간단히 FCC 가정
        elif is_tetragonal:
            return "I4/mmm"
        elif is_orthorhombic:
            return "Pnma"
        else:
            return "P1"

    def _find_symmetry_operations(self, structure: Structure) -> List[str]:
        """대칭 연산 찾기"""
        operations = ['E']  # 항등 연산은 항상 존재

        lattice = np.array(structure.lattice_vectors)
        positions = np.array(structure.positions)

        # 반전 대칭 확인
        if self._has_inversion_symmetry(positions):
            operations.append('i')

        # 미러 대칭 확인
        for plane in ['xy', 'xz', 'yz']:
            if self._has_mirror_symmetry(positions, plane):
                operations.append(f'σ_{plane}')

        return operations

    def _has_inversion_symmetry(self, positions: np.ndarray) -> bool:
        """반전 대칭 확인"""
        center = np.mean(positions, axis=0)
        inverted = 2 * center - positions

        # 반전된 위치가 원래 위치와 일치하는지 확인
        for inv_pos in inverted:
            inv_pos_wrapped = inv_pos % 1.0
            found = False
            for orig_pos in positions:
                orig_pos_wrapped = orig_pos % 1.0
                if np.allclose(inv_pos_wrapped, orig_pos_wrapped, atol=0.1):
                    found = True
                    break
            if not found:
                return False
        return True

    def _has_mirror_symmetry(self, positions: np.ndarray, plane: str) -> bool:
        """미러 대칭 확인"""
        axis_map = {'xy': 2, 'xz': 1, 'yz': 0}
        axis = axis_map[plane]

        mirrored = positions.copy()
        mirrored[:, axis] = -mirrored[:, axis]
        mirrored = mirrored % 1.0

        for mir_pos in mirrored:
            found = False
            for orig_pos in positions:
                orig_pos_wrapped = orig_pos % 1.0
                if np.allclose(mir_pos, orig_pos_wrapped, atol=0.1):
                    found = True
                    break
            if not found:
                return False
        return True

    def _check_equivalent_sites(self, structure: Structure) -> Dict[str, List[int]]:
        """등가 위치 확인"""
        equivalent_sites = {}

        for i, z in enumerate(structure.atomic_numbers):
            z_int = int(z)
            symbol = ELEMENT_SYMBOLS.get(z_int, f'X{z_int}')
            if symbol not in equivalent_sites:
                equivalent_sites[symbol] = []
            equivalent_sites[symbol].append(i)

        return equivalent_sites

    def _analyze_density(self, structure: Structure) -> float:
        """밀도 분석"""
        from utils.constants import ATOMIC_MASS

        lattice = np.array(structure.lattice_vectors)
        volume = abs(np.linalg.det(lattice))  # Å³

        # 총 질량 계산 (amu)
        total_mass = sum(ATOMIC_MASS.get(int(z), 1.0) for z in structure.atomic_numbers)

        # 밀도 계산 (g/cm³)
        # 1 amu = 1.66054e-24 g, 1 Å³ = 1e-24 cm³
        density = total_mass * 1.66054 / volume

        # 일반적인 고체 밀도 범위 (0.5-25 g/cm³)에서 점수 계산
        if 0.5 < density < 25.0:
            return 1.0 - abs(density - 5.0) / 20.0  # 5 g/cm³ 근처에서 최대
        return 0.0

    def _analyze_packing(self, structure: Structure) -> float:
        """패킹 효율 분석"""
        lattice = np.array(structure.lattice_vectors)
        volume = abs(np.linalg.det(lattice))

        # 원자들의 총 부피 계산
        total_atomic_volume = 0.0
        for z in structure.atomic_numbers:
            radius = ATOMIC_RADIUS.get(int(z), 1.5)  # Å
            atomic_volume = (4.0 / 3.0) * np.pi * radius ** 3
            total_atomic_volume += atomic_volume

        # 패킹 비율
        packing_fraction = total_atomic_volume / volume

        # 일반적인 패킹 비율 (0.3-0.74)에서 점수 계산
        if 0.1 < packing_fraction < 0.9:
            # 0.5 근처에서 최대 점수
            return 1.0 - abs(packing_fraction - 0.5) / 0.5
        return 0.0