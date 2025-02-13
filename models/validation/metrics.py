from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error, r2_score

from ...interfaces import Structure, DFTResult, ValidationResult


@dataclass
class ValidationMetrics:
    """검증 메트릭스"""
    # 구조 메트릭스
    rmsd: float  # Root Mean Square Deviation
    max_force: float  # 최대 원자간 힘
    volume_change: float  # 부피 변화율
    symmetry_deviation: float  # 대칭성 편차

    # 에너지 메트릭스
    energy_error: float  # 에너지 오차
    formation_energy_error: float  # 형성 에너지 오차
    energy_per_atom_error: float  # 원자당 에너지 오차

    # 전자 구조 메트릭스
    band_gap_error: float  # 밴드갭 오차
    dos_distance: float  # DOS 분포 거리
    charge_deviation: float  # 전하 분포 편차

    # 종합 평가
    overall_score: float  # 종합 점수
    reliability_score: float  # 신뢰도 점수


class MetricsCalculator:
    """검증 메트릭스 계산기"""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'rmsd': 0.2,
            'max_force': 0.15,
            'volume_change': 0.1,
            'symmetry_deviation': 0.1,
            'energy_error': 0.15,
            'formation_energy_error': 0.1,
            'band_gap_error': 0.1,
            'dos_distance': 0.05,
            'charge_deviation': 0.05
        }

    def calculate_metrics(self,
                          predicted: Structure,
                          actual: Structure,
                          dft_result: Optional[DFTResult] = None) -> ValidationMetrics:
        """메트릭스 계산"""
        # 구조 메트릭스
        structure_metrics = self._calculate_structure_metrics(predicted, actual)

        # DFT 결과가 있는 경우 추가 메트릭스 계산
        if dft_result is not None:
            energy_metrics = self._calculate_energy_metrics(predicted, dft_result)
            electronic_metrics = self._calculate_electronic_metrics(predicted, dft_result)
        else:
            energy_metrics = self._create_empty_energy_metrics()
            electronic_metrics = self._create_empty_electronic_metrics()

        # 종합 점수 계산
        overall_score = self._calculate_overall_score(
            structure_metrics,
            energy_metrics,
            electronic_metrics
        )

        # 신뢰도 점수 계산
        reliability_score = self._calculate_reliability_score(
            structure_metrics,
            energy_metrics,
            electronic_metrics
        )

        return ValidationMetrics(
            **structure_metrics,
            **energy_metrics,
            **electronic_metrics,
            overall_score=overall_score,
            reliability_score=reliability_score
        )

    def _calculate_structure_metrics(self,
                                     predicted: Structure,
                                     actual: Structure) -> Dict[str, float]:
        """구조 메트릭스 계산"""
        # RMSD 계산
        rmsd = self._calculate_rmsd(
            predicted.positions,
            actual.positions,
            predicted.lattice_vectors
        )

        # 최대 힘 계산
        max_force = self._calculate_max_force(predicted, actual)

        # 부피 변화 계산
        volume_change = self._calculate_volume_change(predicted, actual)

        # 대칭성 편차 계산
        symmetry_deviation = self._calculate_symmetry_deviation(predicted, actual)

        return {
            'rmsd': rmsd,
            'max_force': max_force,
            'volume_change': volume_change,
            'symmetry_deviation': symmetry_deviation
        }

    def _calculate_energy_metrics(self,
                                  predicted: Structure,
                                  dft_result: DFTResult) -> Dict[str, float]:
        """에너지 메트릭스 계산"""
        # 실제 에너지값과 비교
        energy_error = abs(predicted.energy - dft_result.total_energy)

        # 형성 에너지 오차
        formation_energy_error = abs(
            predicted.formation_energy - dft_result.formation_energy
        )

        # 원자당 에너지 오차
        energy_per_atom_error = abs(
            predicted.energy / len(predicted.atomic_numbers) -
            dft_result.energy_per_atom
        )

        return {
            'energy_error': energy_error,
            'formation_energy_error': formation_energy_error,
            'energy_per_atom_error': energy_per_atom_error
        }

    def _calculate_electronic_metrics(self,
                                      predicted: Structure,
                                      dft_result: DFTResult) -> Dict[str, float]:
        """전자 구조 메트릭스 계산"""
        # 밴드갭 오차
        band_gap_error = abs(
            predicted.band_gap - dft_result.band_gap
        ) if hasattr(predicted, 'band_gap') else float('inf')

        # DOS 거리
        dos_distance = self._calculate_dos_distance(
            predicted, dft_result
        ) if hasattr(predicted, 'dos') else float('inf')

        # 전하 편차
        charge_deviation = self._calculate_charge_deviation(
            predicted, dft_result
        ) if hasattr(predicted, 'charges') else float('inf')

        return {
            'band_gap_error': band_gap_error,
            'dos_distance': dos_distance,
            'charge_deviation': charge_deviation
        }

    def _calculate_rmsd(self,
                        pos1: np.ndarray,
                        pos2: np.ndarray,
                        lattice: np.ndarray) -> float:
        """RMSD 계산"""
        # 최소 이미지 규약 적용
        diff = pos1 - pos2
        diff = diff - np.round(diff)
        cart_diff = np.dot(diff, lattice)
        return np.sqrt(np.mean(np.sum(cart_diff ** 2, axis=1)))

    def _calculate_dos_distance(self,
                                predicted: Structure,
                                dft_result: DFTResult) -> float:
        """DOS 분포 거리 계산"""
        if not (hasattr(predicted, 'dos') and dft_result.dos is not None):
            return float('inf')

        # Wasserstein 거리 사용
        return wasserstein_distance(
            predicted.dos['energies'],
            dft_result.dos['energies'],
            predicted.dos['values'],
            dft_result.dos['values']
        )

    def _calculate_overall_score(self,
                                 structure_metrics: Dict[str, float],
                                 energy_metrics: Dict[str, float],
                                 electronic_metrics: Dict[str, float]) -> float:
        """종합 점수 계산"""
        total_score = 0.0
        total_weight = 0.0

        for metric_name, value in {
            **structure_metrics,
            **energy_metrics,
            **electronic_metrics
        }.items():
            if metric_name in self.weights and value != float('inf'):
                weight = self.weights[metric_name]
                score = np.exp(-value)  # 오차를 0-1 점수로 변환
                total_score += weight * score
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0