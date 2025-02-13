from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
import random
from enum import Enum

from interfaces import Structure, MutationResult
from protocols import IMutationGenerator
from utils.logger import get_logger


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
        mutated = structure.copy()

        # 무작위 원자 선택
        atom_idx = random.randrange(len(structure.atomic_numbers))

        # 변위 생성
        displacement = np.random.uniform(
            low=self.params.displacement_range[0],
            high=self.params.displacement_range[1],
            size=3
        )

        # 변위 적용
        mutated.positions[atom_idx] += displacement

        return mutated

    def _apply_strain(self, structure: Structure) -> Structure:
        """격자 변형"""
        mutated = structure.copy()

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
            structure.lattice_vectors,
            deformation
        )

        return mutated

    def _apply_rotation(self, structure: Structure) -> Structure:
        """회전"""
        mutated = structure.copy()

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
        mutated.lattice_vectors = np.dot(structure.lattice_vectors, R)
        mutated.positions = np.dot(structure.positions, R)

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
        return {
            'position_changes': np.linalg.norm(
                mutated.positions - original.positions, axis=1
            ),
            'cell_deformation': np.linalg.norm(
                mutated.lattice_vectors - original.lattice_vectors
            ),
            'volume_change': (
                    self._calculate_volume(mutated) /
                    self._calculate_volume(original) - 1
            )
        }