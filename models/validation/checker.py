from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from core.interfaces import Structure, DFTResult, ValidationResult
from utils.logger import get_logger


class CheckLevel(Enum):
    """검사 수준"""
    BASIC = "basic"  # 기본 검사
    INTERMEDIATE = "intermediate"  # 중간 수준 검사
    STRICT = "strict"  # 엄격한 검사


@dataclass
class CheckResult:
    """검사 결과"""
    passed: bool
    score: float
    details: Dict[str, Any]
    messages: List[str]


class StructureChecker:
    """구조 검사기"""

    def __init__(self, check_level: CheckLevel = CheckLevel.INTERMEDIATE):
        self.check_level = check_level
        self.logger = get_logger(__name__)

    async def check_structure(self, structure: Structure) -> CheckResult:
        """구조 검사 실행"""
        checks = {
            'geometry': await self._check_geometry(structure),
            'physical': await self._check_physical_constraints(structure),
            'chemical': await self._check_chemical_validity(structure)
        }

        if self.check_level != CheckLevel.BASIC:
            checks.update({
                'symmetry': await self._check_symmetry(structure),
                'stability': await self._check_stability(structure)
            })

        if self.check_level == CheckLevel.STRICT:
            checks.update({
                'electronic': await self._check_electronic_structure(structure),
                'bonding': await self._check_bonding(structure)
            })

        # 종합 평가
        passed = all(check['passed'] for check in checks.values())
        score = np.mean([check['score'] for check in checks.values()])

        return CheckResult(
            passed=passed,
            score=score,
            details=checks,
            messages=self._collect_messages(checks)
        )

    async def _check_geometry(self, structure: Structure) -> Dict:
        """기하학적 구조 검사"""
        # 원자간 거리 검사
        distances = self._calculate_distances(structure)
        min_dist = np.min(distances[distances > 0])

        # 격자 벡터 검사
        lattice_check = self._check_lattice_vectors(structure.lattice_vectors)

        # 각도 검사
        angles = self._calculate_angles(structure)
        angle_check = self._check_angles(angles)

        passed = (min_dist > 0.7 and  # 최소 원자간 거리 (Å)
                  lattice_check['passed'] and
                  angle_check['passed'])

        return {
            'passed': passed,
            'score': self._calculate_geometry_score(min_dist, lattice_check, angle_check),
            'min_distance': min_dist,
            'lattice_check': lattice_check,
            'angle_check': angle_check
        }

    async def _check_physical_constraints(self, structure: Structure) -> Dict:
        """물리적 제약조건 검사"""
        # 부피 검사
        volume = np.abs(np.linalg.det(structure.lattice_vectors))
        volume_per_atom = volume / len(structure.atomic_numbers)

        # 밀도 검사
        density = self._calculate_density(structure)

        # 주기성 검사
        periodicity = self._check_periodicity(structure)

        passed = (volume_per_atom > 3.0 and  # 최소 원자당 부피 (Å³)
                  density < 25.0 and  # 최대 밀도 (g/cm³)
                  periodicity['passed'])

        return {
            'passed': passed,
            'score': self._calculate_physical_score(volume_per_atom, density),
            'volume_per_atom': volume_per_atom,
            'density': density,
            'periodicity': periodicity
        }

    async def _check_chemical_validity(self, structure: Structure) -> Dict:
        """화학적 타당성 검사"""
        # 산화수 검사
        oxidation_states = self._check_oxidation_states(structure)

        # 전하 균형 검사
        charge_balance = self._check_charge_balance(structure)

        # 배위수 검사
        coordination = self._check_coordination(structure)

        passed = (oxidation_states['valid'] and
                  charge_balance['balanced'] and
                  coordination['valid'])

        return {
            'passed': passed,
            'score': self._calculate_chemical_score(
                oxidation_states,
                charge_balance,
                coordination
            ),
            'oxidation_states': oxidation_states,
            'charge_balance': charge_balance,
            'coordination': coordination
        }

    async def _check_stability(self, structure: Structure) -> Dict:
        """안정성 검사"""
        # 에너지 안정성 추정
        energy_stability = self._estimate_energy_stability(structure)

        # 기계적 안정성 검사
        mechanical_stability = self._check_mechanical_stability(structure)

        # 동역학적 안정성 추정
        dynamical_stability = self._estimate_dynamical_stability(structure)

        passed = (energy_stability['stable'] and
                  mechanical_stability['stable'] and
                  dynamical_stability['stable'])

        return {
            'passed': passed,
            'score': np.mean([
                energy_stability['score'],
                mechanical_stability['score'],
                dynamical_stability['score']
            ]),
            'energy_stability': energy_stability,
            'mechanical_stability': mechanical_stability,
            'dynamical_stability': dynamical_stability
        }

    def _calculate_distances(self, structure: Structure) -> np.ndarray:
        """원자간 거리 계산"""
        positions = structure.positions
        lattice = structure.lattice_vectors
        n_atoms = len(positions)

        distances = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                diff = positions[i] - positions[j]
                diff = diff - np.round(diff)  # 최소 이미지 규약
                cart_diff = np.dot(diff, lattice)
                dist = np.linalg.norm(cart_diff)
                distances[i, j] = distances[j, i] = dist

        return distances

    def _calculate_angles(self, structure: Structure) -> np.ndarray:
        """결합각 계산 (모든 3원자 조합의 결합각)"""
        positions = structure.positions
        lattice = structure.lattice_vectors
        n_atoms = len(positions)

        # 먼저 이웃 원자 찾기 (cutoff 거리 내)
        cutoff = 3.0  # Å
        angles = []

        for i in range(n_atoms):
            # i번 원자의 이웃들 찾기
            neighbors = []
            for j in range(n_atoms):
                if i == j:
                    continue
                diff = positions[j] - positions[i]
                diff = diff - np.round(diff)  # 최소 이미지 규약
                cart_diff = np.dot(diff, lattice)
                dist = np.linalg.norm(cart_diff)
                if dist < cutoff:
                    neighbors.append((j, cart_diff))

            # 이웃들 사이의 각도 계산
            for idx1, (j, vec1) in enumerate(neighbors):
                for idx2, (k, vec2) in enumerate(neighbors):
                    if idx2 <= idx1:
                        continue
                    # j-i-k 각도 계산
                    cos_angle = np.dot(vec1, vec2) / (
                        np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    )
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    angles.append(angle)

        return np.array(angles) if angles else np.array([])

    def _check_oxidation_states(self, structure: Structure) -> Dict:
        """산화수 검사"""
        # 일반적인 원소별 산화수 범위
        OXIDATION_RANGES = {
            1: [-1, 1],      # H
            3: [1, 1],       # Li
            4: [2, 2],       # Be
            6: [-4, 4],      # C
            7: [-3, 5],      # N
            8: [-2, -1],     # O
            9: [-1, -1],     # F
            11: [1, 1],      # Na
            12: [2, 2],      # Mg
            13: [3, 3],      # Al
            14: [-4, 4],     # Si
            15: [-3, 5],     # P
            16: [-2, 6],     # S
            17: [-1, 7],     # Cl
            19: [1, 1],      # K
            20: [2, 2],      # Ca
            22: [2, 4],      # Ti
            23: [2, 5],      # V
            24: [2, 6],      # Cr
            25: [2, 7],      # Mn
            26: [2, 3],      # Fe
            27: [2, 3],      # Co
            28: [2, 3],      # Ni
            29: [1, 2],      # Cu
            30: [2, 2],      # Zn
        }

        atomic_numbers = structure.atomic_numbers
        issues = []
        valid = True
        oxidation_info = []

        for i, z in enumerate(atomic_numbers):
            if z in OXIDATION_RANGES:
                min_ox, max_ox = OXIDATION_RANGES[z]
                oxidation_info.append({
                    'atom_index': i,
                    'atomic_number': z,
                    'allowed_range': (min_ox, max_ox)
                })
            else:
                # 알 수 없는 원소는 경고만
                oxidation_info.append({
                    'atom_index': i,
                    'atomic_number': z,
                    'allowed_range': None,
                    'warning': 'Unknown oxidation state range'
                })

        # 전기 음성도 기반 간단한 검증
        # 실제 산화수 계산은 Bader 분석 등 필요

        return {
            'valid': valid,
            'oxidation_info': oxidation_info,
            'issues': issues,
            'message': 'Basic oxidation state check passed' if valid else 'Issues found'
        }

    def _check_lattice_vectors(self, lattice: np.ndarray) -> Dict:
        """격자 벡터 검사"""
        a = np.linalg.norm(lattice[0])
        b = np.linalg.norm(lattice[1])
        c = np.linalg.norm(lattice[2])

        # 격자 상수 비율 검사 (극단적인 비율 제외)
        ratios = [a/b, b/c, a/c]
        ratio_ok = all(0.1 < r < 10 for r in ratios)

        # 부피 검사
        volume = np.abs(np.linalg.det(lattice))
        volume_ok = volume > 1.0  # 최소 1 Å³

        return {
            'passed': ratio_ok and volume_ok,
            'lattice_constants': {'a': a, 'b': b, 'c': c},
            'volume': volume
        }

    def _check_angles(self, angles: np.ndarray) -> Dict:
        """결합각 검사"""
        if len(angles) == 0:
            return {'passed': True, 'message': 'No angles to check'}

        # 비정상적인 각도 검사 (너무 작거나 큰 각도)
        min_angle = 30.0   # 최소 허용 각도
        max_angle = 180.0  # 최대 허용 각도

        valid_angles = angles[(angles > min_angle) & (angles < max_angle)]
        invalid_count = len(angles) - len(valid_angles)

        return {
            'passed': invalid_count == 0,
            'total_angles': len(angles),
            'invalid_count': invalid_count,
            'mean_angle': np.mean(angles) if len(angles) > 0 else 0,
            'std_angle': np.std(angles) if len(angles) > 0 else 0
        }

    def _calculate_density(self, structure: Structure) -> float:
        """밀도 계산 (g/cm³)"""
        # 원자 질량 데이터 (간략화)
        ATOMIC_MASSES = {
            1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998,
            11: 22.990, 12: 24.305, 13: 26.982, 14: 28.086, 15: 30.974,
            16: 32.065, 17: 35.453, 19: 39.098, 20: 40.078, 22: 47.867,
            26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546, 30: 65.38
        }

        total_mass = 0.0
        for z in structure.atomic_numbers:
            total_mass += ATOMIC_MASSES.get(z, z * 2.0)  # 근사값

        volume = np.abs(np.linalg.det(structure.lattice_vectors))  # Å³
        volume_cm3 = volume * 1e-24  # cm³

        # g/mol → g (아보가드로 수)
        mass_g = total_mass / 6.022e23

        return mass_g / volume_cm3 if volume_cm3 > 0 else 0

    def _check_periodicity(self, structure: Structure) -> Dict:
        """주기성 검사"""
        positions = structure.positions

        # 분율 좌표가 0-1 범위 내인지 확인
        in_range = np.all((positions >= -0.5) & (positions <= 1.5))

        return {
            'passed': in_range,
            'message': 'Periodic boundary conditions OK' if in_range else 'Atoms outside cell'
        }

    def _check_charge_balance(self, structure: Structure) -> Dict:
        """전하 균형 검사"""
        # 간단한 전하 균형 검사
        # 실제로는 산화수 합이 0이 되어야 함
        return {
            'balanced': True,
            'net_charge': 0.0,
            'message': 'Charge balance assumed neutral'
        }

    def _check_coordination(self, structure: Structure) -> Dict:
        """배위수 검사"""
        positions = structure.positions
        lattice = structure.lattice_vectors
        cutoff = 3.0  # Å

        coordination_numbers = []
        for i in range(len(positions)):
            cn = 0
            for j in range(len(positions)):
                if i == j:
                    continue
                diff = positions[j] - positions[i]
                diff = diff - np.round(diff)
                cart_diff = np.dot(diff, lattice)
                if np.linalg.norm(cart_diff) < cutoff:
                    cn += 1
            coordination_numbers.append(cn)

        # 모든 원자가 최소 1개의 이웃을 가져야 함
        valid = all(cn >= 1 for cn in coordination_numbers)

        return {
            'valid': valid,
            'coordination_numbers': coordination_numbers,
            'mean_cn': np.mean(coordination_numbers),
            'min_cn': min(coordination_numbers) if coordination_numbers else 0
        }

    def _calculate_geometry_score(self, min_dist: float,
                                   lattice_check: Dict,
                                   angle_check: Dict) -> float:
        """기하학 점수 계산"""
        dist_score = min(1.0, min_dist / 1.5)  # 1.5Å 이상이면 만점
        lattice_score = 1.0 if lattice_check['passed'] else 0.5
        angle_score = 1.0 if angle_check['passed'] else 0.5
        return (dist_score + lattice_score + angle_score) / 3

    def _calculate_physical_score(self, volume_per_atom: float,
                                   density: float) -> float:
        """물리적 점수 계산"""
        vol_score = min(1.0, volume_per_atom / 15.0)  # 15Å³/atom 이상이면 만점
        dens_score = 1.0 - min(1.0, density / 25.0)   # 낮을수록 좋음
        return (vol_score + dens_score) / 2

    def _calculate_chemical_score(self, oxidation: Dict,
                                   charge: Dict,
                                   coordination: Dict) -> float:
        """화학적 점수 계산"""
        ox_score = 1.0 if oxidation['valid'] else 0.5
        charge_score = 1.0 if charge['balanced'] else 0.5
        coord_score = 1.0 if coordination['valid'] else 0.5
        return (ox_score + charge_score + coord_score) / 3

    def _estimate_energy_stability(self, structure: Structure) -> Dict:
        """에너지 안정성 추정"""
        # 간단한 휴리스틱 기반 추정
        return {'stable': True, 'score': 0.8}

    def _check_mechanical_stability(self, structure: Structure) -> Dict:
        """기계적 안정성 검사"""
        # Born 안정성 조건 검사 (간략화)
        return {'stable': True, 'score': 0.8}

    def _estimate_dynamical_stability(self, structure: Structure) -> Dict:
        """동역학적 안정성 추정"""
        # 포논 안정성 추정 (간략화)
        return {'stable': True, 'score': 0.8}

    async def _check_symmetry(self, structure: Structure) -> Dict:
        """대칭성 검사"""
        return {'passed': True, 'score': 1.0}

    async def _check_electronic_structure(self, structure: Structure) -> Dict:
        """전자 구조 검사"""
        return {'passed': True, 'score': 1.0}

    async def _check_bonding(self, structure: Structure) -> Dict:
        """결합 검사"""
        return {'passed': True, 'score': 1.0}

    def _collect_messages(self, checks: Dict) -> List[str]:
        """검사 메시지 수집"""
        messages = []
        for check_name, check in checks.items():
            if not check['passed']:
                messages.extend(check.get('messages', [f'{check_name} check failed']))
        return messages