from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass, replace
import random
from enum import Enum
import copy

from core.interfaces import Structure, MutationResult
from core.protocols import IMutationGenerator
from utils.logger import get_logger
from utils.constants import (
    ELEMENT_SYMBOLS, ATOMIC_NUMBERS, ELECTRONEGATIVITY,
    MIN_ATOMIC_DISTANCE, MAX_ATOMIC_DISTANCE, MAX_VOLUME_CHANGE
)


class MutationType(Enum):
    """Mutation 유형"""
    ATOMIC_DISPLACEMENT = "displacement"
    ATOMIC_SUBSTITUTION = "substitution"
    ATOMIC_ADDITION = "addition"
    ATOMIC_REMOVAL = "removal"
    LATTICE_STRAIN = "strain"
    LATTICE_ROTATION = "rotation"
    COORDINATION_CHANGE = "coordination"
    SYMMETRY_OPERATION = "symmetry"


@dataclass
class MutationParameters:
    """Mutation 파라미터"""
    displacement_range: Tuple[float, float] = (0.1, 0.5)  # Å
    strain_range: Tuple[float, float] = (-0.1, 0.1)  # 상대값
    rotation_range: Tuple[float, float] = (-30.0, 30.0)  # degrees
    probabilities: Dict[MutationType, float] = None
    max_attempts: int = 100

    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = {
                MutationType.ATOMIC_DISPLACEMENT: 0.3,
                MutationType.ATOMIC_SUBSTITUTION: 0.1,
                MutationType.ATOMIC_ADDITION: 0.1,
                MutationType.ATOMIC_REMOVAL: 0.1,
                MutationType.LATTICE_STRAIN: 0.2,
                MutationType.LATTICE_ROTATION: 0.1,
                MutationType.COORDINATION_CHANGE: 0.05,
                MutationType.SYMMETRY_OPERATION: 0.05
            }


class MutationGenerator(IMutationGenerator):
    """Mutation 생성기"""

    def __init__(self, params: Optional[MutationParameters] = None):
        self.params = params or MutationParameters()
        self.logger = get_logger(__name__)

    async def generate(self,
                       structure: Structure,
                       n_mutations: int = 1) -> List[MutationResult]:
        """Mutation 생성"""
        results = []
        attempts = 0

        while len(results) < n_mutations and attempts < self.params.max_attempts:
            try:
                # Mutation 유형 선택
                mutation_type = self._select_mutation_type()

                # Mutation 적용
                mutated = self._apply_mutation(structure, mutation_type)

                # 유효성 검사
                if await self.validate_mutation(structure, mutated):
                    # 품질 평가
                    quality = await self.estimate_quality(mutated)

                    results.append(MutationResult(
                        original_structure=structure,
                        mutated_structure=mutated,
                        mutation_type=mutation_type.value,
                        changes=self._describe_changes(structure, mutated),
                        success=True,
                        stability_score=quality['stability'],
                        validity_score=quality['validity'],
                        energy_estimate=quality['energy']
                    ))

            except Exception as e:
                self.logger.warning(f"Mutation failed: {e}")

            attempts += 1

        return results

    def _select_mutation_type(self) -> MutationType:
        """Mutation 유형 선택"""
        return random.choices(
            list(self.params.probabilities.keys()),
            weights=list(self.params.probabilities.values())
        )[0]

    def _apply_mutation(self,
                        structure: Structure,
                        mutation_type: MutationType) -> Structure:
        """Mutation 적용"""
        if mutation_type == MutationType.ATOMIC_DISPLACEMENT:
            return self._apply_displacement(structure)
        elif mutation_type == MutationType.ATOMIC_SUBSTITUTION:
            return self._apply_substitution(structure)
        elif mutation_type == MutationType.ATOMIC_ADDITION:
            return self._apply_addition(structure)
        elif mutation_type == MutationType.ATOMIC_REMOVAL:
            return self._apply_removal(structure)
        elif mutation_type == MutationType.LATTICE_STRAIN:
            return self._apply_strain(structure)
        elif mutation_type == MutationType.LATTICE_ROTATION:
            return self._apply_rotation(structure)
        elif mutation_type == MutationType.COORDINATION_CHANGE:
            return self._apply_coordination_change(structure)
        elif mutation_type == MutationType.SYMMETRY_OPERATION:
            return self._apply_symmetry_operation(structure)

    def _apply_displacement(self, structure: Structure) -> Structure:
        """원자 변위"""
        mutated = self._copy_structure(structure)

        # 무작위 원자 선택
        atom_idx = random.randrange(len(structure.atomic_numbers))

        # 변위 생성
        displacement = np.random.uniform(
            low=self.params.displacement_range[0],
            high=self.params.displacement_range[1],
            size=3
        )

        # 변위 적용 (분율 좌표이므로 격자로 나눠서 정규화)
        lattice = np.array(structure.lattice_vectors)
        frac_displacement = np.linalg.solve(lattice.T, displacement)
        mutated.positions[atom_idx] += frac_displacement

        return mutated

    def _apply_strain(self, structure: Structure) -> Structure:
        """격자 변형"""
        mutated = self._copy_structure(structure)

        # 변형 텐서 생성
        strain = np.random.uniform(
            low=self.params.strain_range[0],
            high=self.params.strain_range[1],
            size=(3, 3)
        )
        strain = (strain + strain.T) / 2  # 대칭화

        # 변형 적용
        identity = np.eye(3)
        deformation = identity + strain
        mutated.lattice_vectors = np.dot(
            np.array(structure.lattice_vectors),
            deformation
        )

        return mutated

    def _apply_rotation(self, structure: Structure) -> Structure:
        """회전"""
        mutated = self._copy_structure(structure)

        # 회전 각도 및 축 선택
        angle = np.random.uniform(
            low=self.params.rotation_range[0],
            high=self.params.rotation_range[1]
        )
        axis = np.random.rand(3)
        axis /= np.linalg.norm(axis)

        # 회전 행렬 생성
        theta = np.radians(angle)
        R = self._rotation_matrix(axis, theta)

        # 회전 적용
        mutated.lattice_vectors = np.dot(np.array(structure.lattice_vectors), R)
        mutated.positions = np.dot(np.array(structure.positions), R)

        return mutated

    def _rotation_matrix(self, axis: np.ndarray, theta: float) -> np.ndarray:
        """회전 행렬 계산 (Rodrigues 회전 공식)"""
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)

        return np.array([
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]
        ])

    async def validate_mutation(self,
                                original: Structure,
                                mutated: Structure) -> bool:
        """Mutation 유효성 검사"""
        return all([
            self._check_atomic_distances(mutated),
            self._check_cell_volume(mutated),
            self._check_coordination(mutated)
        ])

    async def estimate_quality(self,
                               mutated: Structure) -> Dict[str, float]:
        """Mutation 품질 추정"""
        return {
            'stability': self._estimate_stability(mutated),
            'validity': self._estimate_validity(mutated),
            'energy': self._estimate_energy(mutated)
        }

    def _describe_changes(self,
                          original: Structure,
                          mutated: Structure) -> Dict:
        """구조 변화 설명"""
        orig_pos = np.array(original.positions)
        mut_pos = np.array(mutated.positions)

        # 원자 수가 다른 경우 처리
        if orig_pos.shape != mut_pos.shape:
            return {
                'n_atoms_change': len(mutated.atomic_numbers) - len(original.atomic_numbers),
                'volume_change': self._calculate_volume(mutated) / self._calculate_volume(original) - 1
            }

        return {
            'position_changes': float(np.mean(np.linalg.norm(mut_pos - orig_pos, axis=1))),
            'cell_deformation': float(np.linalg.norm(
                np.array(mutated.lattice_vectors) - np.array(original.lattice_vectors)
            )),
            'volume_change': self._calculate_volume(mutated) / self._calculate_volume(original) - 1
        }

    def _copy_structure(self, structure: Structure) -> Structure:
        """Structure 복사"""
        return Structure(
            atomic_numbers=np.array(structure.atomic_numbers).copy(),
            positions=np.array(structure.positions).copy(),
            lattice_vectors=np.array(structure.lattice_vectors).copy(),
            cell_params=dict(structure.cell_params),
            formula=structure.formula,
            space_group=structure.space_group,
            charge=structure.charge,
            spin=structure.spin,
            constraints=copy.deepcopy(structure.constraints) if structure.constraints else None
        )

    def _apply_substitution(self, structure: Structure) -> Structure:
        """원소 치환"""
        mutated = self._copy_structure(structure)

        # 무작위 원자 선택
        atom_idx = random.randrange(len(structure.atomic_numbers))
        old_z = int(mutated.atomic_numbers[atom_idx])

        # 유사한 원소로 치환 (주기율표에서 근처)
        candidates = [z for z in ELEMENT_SYMBOLS.keys() if abs(z - old_z) <= 10 and z != old_z]
        if candidates:
            new_z = random.choice(candidates)
            mutated.atomic_numbers[atom_idx] = new_z
            mutated.formula = self._generate_formula(mutated.atomic_numbers)

        return mutated

    def _apply_addition(self, structure: Structure) -> Structure:
        """원자 추가"""
        mutated = self._copy_structure(structure)

        # 기존 원자 중 하나와 같은 종류 추가
        z_to_add = random.choice(list(mutated.atomic_numbers))

        # 무작위 위치에 추가 (기존 원자 근처)
        existing_pos = random.choice(mutated.positions)
        new_pos = existing_pos + np.random.uniform(-0.3, 0.3, size=3)
        new_pos = new_pos % 1.0  # 주기적 경계 조건

        mutated.atomic_numbers = np.append(mutated.atomic_numbers, z_to_add)
        mutated.positions = np.vstack([mutated.positions, new_pos])
        mutated.formula = self._generate_formula(mutated.atomic_numbers)

        return mutated

    def _apply_removal(self, structure: Structure) -> Structure:
        """원자 제거"""
        if len(structure.atomic_numbers) <= 1:
            return structure  # 최소 1개는 유지

        mutated = self._copy_structure(structure)

        # 무작위 원자 선택하여 제거
        atom_idx = random.randrange(len(mutated.atomic_numbers))

        mutated.atomic_numbers = np.delete(mutated.atomic_numbers, atom_idx)
        mutated.positions = np.delete(mutated.positions, atom_idx, axis=0)
        mutated.formula = self._generate_formula(mutated.atomic_numbers)

        return mutated

    def _apply_coordination_change(self, structure: Structure) -> Structure:
        """배위수 변경 (원자 이동으로 구현)"""
        mutated = self._copy_structure(structure)

        if len(mutated.atomic_numbers) < 2:
            return mutated

        # 두 원자를 더 가깝게 또는 멀게 이동
        idx1, idx2 = random.sample(range(len(mutated.atomic_numbers)), 2)

        direction = mutated.positions[idx2] - mutated.positions[idx1]
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # 50% 확률로 가깝게 또는 멀게
        factor = random.choice([-0.1, 0.1])
        mutated.positions[idx1] -= direction * factor
        mutated.positions[idx2] += direction * factor

        return mutated

    def _apply_symmetry_operation(self, structure: Structure) -> Structure:
        """대칭 연산 적용"""
        mutated = self._copy_structure(structure)

        # 간단한 대칭 연산: 반전, 회전, 미러
        operation = random.choice(['inversion', 'mirror_xy', 'mirror_xz', 'mirror_yz'])

        if operation == 'inversion':
            center = np.mean(mutated.positions, axis=0)
            mutated.positions = 2 * center - mutated.positions
        elif operation == 'mirror_xy':
            mutated.positions[:, 2] = -mutated.positions[:, 2] + 1.0
        elif operation == 'mirror_xz':
            mutated.positions[:, 1] = -mutated.positions[:, 1] + 1.0
        elif operation == 'mirror_yz':
            mutated.positions[:, 0] = -mutated.positions[:, 0] + 1.0

        # 주기적 경계 조건 적용
        mutated.positions = mutated.positions % 1.0

        return mutated

    def _check_atomic_distances(self, structure: Structure) -> bool:
        """원자간 거리 검사"""
        positions = np.array(structure.positions)
        lattice = np.array(structure.lattice_vectors)
        n_atoms = len(positions)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                diff = positions[j] - positions[i]
                diff = diff - np.round(diff)  # 최소 이미지 규약
                cart_diff = np.dot(diff, lattice)
                distance = np.linalg.norm(cart_diff)

                if distance < MIN_ATOMIC_DISTANCE:
                    return False
                if distance > MAX_ATOMIC_DISTANCE * 2:  # 너무 멀면 결합이 없는 것
                    continue

        return True

    def _check_cell_volume(self, structure: Structure) -> bool:
        """셀 부피 검사"""
        volume = self._calculate_volume(structure)
        n_atoms = len(structure.atomic_numbers)

        # 원자당 최소/최대 부피 검사
        volume_per_atom = volume / n_atoms
        return 5.0 < volume_per_atom < 100.0  # Å³/atom

    def _check_coordination(self, structure: Structure) -> bool:
        """배위수 검사"""
        # 간단한 검사: 모든 원자가 최소 1개 이상의 이웃을 가지는지
        positions = np.array(structure.positions)
        lattice = np.array(structure.lattice_vectors)
        n_atoms = len(positions)

        for i in range(n_atoms):
            has_neighbor = False
            for j in range(n_atoms):
                if i != j:
                    diff = positions[j] - positions[i]
                    diff = diff - np.round(diff)
                    cart_diff = np.dot(diff, lattice)
                    distance = np.linalg.norm(cart_diff)
                    if distance < MAX_ATOMIC_DISTANCE:
                        has_neighbor = True
                        break
            if not has_neighbor and n_atoms > 1:
                return False

        return True

    def _estimate_stability(self, structure: Structure) -> float:
        """안정성 추정"""
        # 간단한 휴리스틱: 원자간 거리 분포 기반
        positions = np.array(structure.positions)
        lattice = np.array(structure.lattice_vectors)
        n_atoms = len(positions)

        if n_atoms < 2:
            return 0.5

        distances = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                diff = positions[j] - positions[i]
                diff = diff - np.round(diff)
                cart_diff = np.dot(diff, lattice)
                distances.append(np.linalg.norm(cart_diff))

        if not distances:
            return 0.5

        # 거리 분산이 작을수록 안정
        std = np.std(distances)
        return float(1.0 / (1.0 + std))

    def _estimate_validity(self, structure: Structure) -> float:
        """유효성 추정"""
        score = 1.0

        # 거리 검사
        if not self._check_atomic_distances(structure):
            score *= 0.5

        # 부피 검사
        if not self._check_cell_volume(structure):
            score *= 0.7

        # 배위수 검사
        if not self._check_coordination(structure):
            score *= 0.8

        return score

    def _estimate_energy(self, structure: Structure) -> float:
        """에너지 추정 (간단한 쌍 포텐셜)"""
        positions = np.array(structure.positions)
        lattice = np.array(structure.lattice_vectors)
        n_atoms = len(positions)

        if n_atoms < 2:
            return 0.0

        energy = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                diff = positions[j] - positions[i]
                diff = diff - np.round(diff)
                cart_diff = np.dot(diff, lattice)
                r = np.linalg.norm(cart_diff)

                if r > 0.1:
                    # 간단한 Lennard-Jones 스타일 포텐셜
                    sigma = 2.5  # Å
                    epsilon = 0.01  # eV
                    energy += 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

        return float(energy)

    def _calculate_volume(self, structure: Structure) -> float:
        """셀 부피 계산"""
        lattice = np.array(structure.lattice_vectors)
        return float(abs(np.linalg.det(lattice)))

    def _generate_formula(self, atomic_numbers) -> str:
        """화학식 생성"""
        from collections import Counter
        counts = Counter(atomic_numbers)
        parts = []
        for z, count in sorted(counts.items()):
            symbol = ELEMENT_SYMBOLS.get(int(z), f'X{z}')
            if count == 1:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}{count}")
        return ''.join(parts)