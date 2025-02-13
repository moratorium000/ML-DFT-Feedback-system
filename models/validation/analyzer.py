from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from scipy import stats
from sklearn.decomposition import PCA
import networkx as nx

from core.interfaces import Structure, DFTResult
from utils.logger import get_logger


@dataclass
class AnalysisResult:
    """분석 결과"""
    structural_analysis: Dict[str, any]  # 구조 분석
    electronic_analysis: Dict[str, any]  # 전자 구조 분석
    energetic_analysis: Dict[str, any]  # 에너지 분석
    statistical_analysis: Dict[str, any]  # 통계 분석
    recommendations: List[str]  # 개선 추천사항


class StructureAnalyzer:
    """구조 분석기"""

    def __init__(self):
        self.logger = get_logger(__name__)

    async def analyze_structure(self,
                                structure: Structure,
                                dft_result: Optional[DFTResult] = None) -> AnalysisResult:
        """구조 분석 실행"""
        # 구조 분석
        structural = await self._analyze_structural_properties(structure)

        # 전자 구조 분석
        electronic = await self._analyze_electronic_properties(structure, dft_result)

        # 에너지 분석
        energetic = await self._analyze_energetic_properties(structure, dft_result)

        # 통계 분석
        statistical = await self._analyze_statistical_properties(
            structure,
            structural,
            electronic,
            energetic
        )

        # 개선 추천사항 생성
        recommendations = self._generate_recommendations(
            structural,
            electronic,
            energetic,
            statistical
        )

        return AnalysisResult(
            structural_analysis=structural,
            electronic_analysis=electronic,
            energetic_analysis=energetic,
            statistical_analysis=statistical,
            recommendations=recommendations
        )

    async def _analyze_structural_properties(self,
                                             structure: Structure) -> Dict[str, any]:
        """구조적 특성 분석"""
        # 격자 분석
        lattice_analysis = self._analyze_lattice(structure)

        # 원자 배열 분석
        atomic_arrangement = self._analyze_atomic_arrangement(structure)

        # 대칭성 분석
        symmetry_analysis = self._analyze_symmetry(structure)

        # 결합 네트워크 분석
        bonding_network = self._analyze_bonding_network(structure)

        return {
            'lattice': lattice_analysis,
            'atomic_arrangement': atomic_arrangement,
            'symmetry': symmetry_analysis,
            'bonding_network': bonding_network,
            'dimensionality': self._analyze_dimensionality(structure)
        }

    async def _analyze_electronic_properties(self,
                                             structure: Structure,
                                             dft_result: Optional[DFTResult]) -> Dict[str, any]:
        """전자 구조 분석"""
        if dft_result is None:
            return {'available': False}

        return {
            'available': True,
            'band_structure': self._analyze_band_structure(dft_result),
            'dos': self._analyze_dos(dft_result),
            'charge_distribution': self._analyze_charge_distribution(dft_result),
            'orbital_analysis': self._analyze_orbitals(dft_result)
        }

    async def _analyze_energetic_properties(self,
                                            structure: Structure,
                                            dft_result: Optional[DFTResult]) -> Dict[str, any]:
        """에너지 특성 분석"""
        if dft_result is None:
            return {'available': False}

        return {
            'available': True,
            'total_energy': self._analyze_total_energy(dft_result),
            'formation_energy': self._analyze_formation_energy(dft_result),
            'cohesive_energy': self._analyze_cohesive_energy(dft_result),
            'energy_decomposition': self._analyze_energy_decomposition(dft_result)
        }

    def _analyze_lattice(self, structure: Structure) -> Dict[str, any]:
        """격자 분석"""
        lattice = structure.lattice_vectors

        # 격자 상수 및 각도
        a, b, c = np.linalg.norm(lattice, axis=1)
        alpha, beta, gamma = self._calculate_angles(lattice)

        # 부피 및 밀도
        volume = np.abs(np.linalg.det(lattice))
        density = self._calculate_density(structure, volume)

        # 격자 변형 분석
        strain_tensor = self._calculate_strain_tensor(lattice)

        return {
            'constants': {'a': a, 'b': b, 'c': c},
            'angles': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
            'volume': volume,
            'density': density,
            'strain': strain_tensor
        }

    def _analyze_atomic_arrangement(self, structure: Structure) -> Dict[str, any]:
        """원자 배열 분석"""
        # 원자 분포 분석
        distribution = self._analyze_atomic_distribution(structure)

        # 배위 환경 분석
        coordination = self._analyze_coordination_environment(structure)

        # 패킹 분석
        packing = self._analyze_packing(structure)

        # 원자 클러스터 분석
        clusters = self._analyze_atomic_clusters(structure)

        return {
            'distribution': distribution,
            'coordination': coordination,
            'packing': packing,
            'clusters': clusters
        }

    def _analyze_bonding_network(self, structure: Structure) -> Dict[str, any]:
        """결합 네트워크 분석"""
        # 결합 그래프 생성
        graph = self._create_bonding_graph(structure)

        # 네트워크 지표 계산
        metrics = {
            'average_degree': np.mean([d for n, d in graph.degree()]),
            'clustering_coefficient': nx.average_clustering(graph),
            'path_lengths': dict(nx.all_pairs_shortest_path_length(graph)),
            'centrality': dict(nx.degree_centrality(graph))
        }

        # 구조적 모티프 분석
        motifs = self._analyze_structural_motifs(graph)

        return {
            'network_metrics': metrics,
            'motifs': motifs,
            'connectivity': nx.is_connected(graph),
            'dimensionality': self._estimate_network_dimensionality(graph)
        }

    def _analyze_statistical_properties(self,
                                        structure: Structure,
                                        structural: Dict,
                                        electronic: Dict,
                                        energetic: Dict) -> Dict[str, any]:
        """통계적 특성 분석"""
        # 구조 특성의 통계 분석
        structural_stats = self._calculate_structural_statistics(structural)

        # 전자 구조 특성의 통계 분석
        electronic_stats = (
            self._calculate_electronic_statistics(electronic)
            if electronic['available'] else None
        )

        # 주성분 분석
        pca_results = self._perform_pca_analysis(
            structure,
            structural,
            electronic,
            energetic
        )

        return {
            'structural_statistics': structural_stats,
            'electronic_statistics': electronic_stats,
            'pca_analysis': pca_results,
            'correlation_analysis': self._analyze_correlations(
                structural,
                electronic,
                energetic
            )
        }

    def _generate_recommendations(self,
                                  structural: Dict,
                                  electronic: Dict,
                                  energetic: Dict,
                                  statistical: Dict) -> List[str]:
        """개선 추천사항 생성"""
        recommendations = []

        # 구조적 개선사항
        if structural['symmetry'].get('broken_symmetry'):
            recommendations.append(
                "Consider improving structural symmetry"
            )

        # 전자 구조 개선사항
        if electronic['available'] and electronic['band_structure'].get('indirect_gap'):
            recommendations.append(
                "Consider modifications to achieve direct band gap"
            )

        # 에너지 관련 개선사항
        if (energetic['available'] and
                energetic['formation_energy'].get('value', 0) > 0):
            recommendations.append(
                "Structure might be metastable, consider stability improvements"
            )

        return recommendations