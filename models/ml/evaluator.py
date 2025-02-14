from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dataclasses import dataclass

from core.interfaces import Structure, PredictionResult, DFTResult, PathStep
from utils.logger import get_logger


@dataclass
class EvaluationMetrics:
    """평가 메트릭스"""
    mse: float
    mae: float
    r2: float
    max_error: float
    calibration_error: float
    confidence_score: float
    property_specific: Dict[str, Dict[str, float]]


class ModelEvaluator:
    """ML 모델 평가기"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)

    async def evaluate_property_predictions(self,
                                            predictions: List[PredictionResult],
                                            actual_results: List[DFTResult]) -> EvaluationMetrics:
        """물성 예측 평가"""
        # 전체 메트릭스
        all_metrics = {}
        property_metrics = {}

        # 각 물성별 평가
        for prop in self.config['target_properties']:
            pred_values = [p.predicted_values[prop] for p in predictions]
            true_values = [r.__dict__[prop] for r in actual_results]
            uncertainties = [p.uncertainty[prop] for p in predictions]

            # 기본 메트릭스 계산
            metrics = self._calculate_basic_metrics(pred_values, true_values)

            # 불확실성 평가
            uncertainty_metrics = self._evaluate_uncertainty(
                pred_values,
                true_values,
                uncertainties
            )

            # 물성별 메트릭스 저장
            property_metrics[prop] = {**metrics, **uncertainty_metrics}

        # 종합 점수 계산
        overall_metrics = self._aggregate_metrics(property_metrics)

        return EvaluationMetrics(
            mse=overall_metrics['mse'],
            mae=overall_metrics['mae'],
            r2=overall_metrics['r2'],
            max_error=overall_metrics['max_error'],
            calibration_error=overall_metrics['calibration_error'],
            confidence_score=overall_metrics['confidence_score'],
            property_specific=property_metrics
        )

    async def evaluate_path_predictions(self,
                                        predicted_paths: List[List[PathStep]],
                                        actual_paths: List[List[PathStep]]) -> Dict[str, float]:
        """경로 예측 평가"""
        path_metrics = {}

        # 경로 길이 평가
        length_diff = self._evaluate_path_lengths(predicted_paths, actual_paths)
        path_metrics['length_difference'] = length_diff

        # 구조 유사도 평가
        structure_similarity = self._evaluate_structure_similarity(
            predicted_paths,
            actual_paths
        )
        path_metrics['structure_similarity'] = structure_similarity

        # 에너지 프로파일 평가
        energy_profile_error = self._evaluate_energy_profiles(
            predicted_paths,
            actual_paths
        )
        path_metrics['energy_profile_error'] = energy_profile_error

        # 경로 연속성 평가
        continuity_score = self._evaluate_path_continuity(predicted_paths)
        path_metrics['continuity_score'] = continuity_score

        return path_metrics

    def _calculate_basic_metrics(self,
                                 predictions: np.ndarray,
                                 true_values: np.ndarray) -> Dict[str, float]:
        """기본 메트릭스 계산"""
        return {
            'mse': mean_squared_error(true_values, predictions),
            'mae': mean_absolute_error(true_values, predictions),
            'r2': r2_score(true_values, predictions),
            'max_error': np.max(np.abs(predictions - true_values))
        }

    def _evaluate_uncertainty(self,
                              predictions: np.ndarray,
                              true_values: np.ndarray,
                              uncertainties: np.ndarray) -> Dict[str, float]:
        """불확실성 평가"""
        # 신뢰 구간 계산
        z_score = stats.norm.ppf(0.95)
        lower = predictions - z_score * uncertainties
        upper = predictions + z_score * uncertainties

        # 보정 오차 계산
        in_interval = np.logical_and(
            true_values >= lower,
            true_values <= upper
        )
        calibration_error = np.abs(np.mean(in_interval) - 0.95)

        # 불확실성-오차 상관관계
        error_uncertainty_corr = np.corrcoef(
            np.abs(predictions - true_values),
            uncertainties
        )[0, 1]

        return {
            'calibration_error': calibration_error,
            'uncertainty_correlation': error_uncertainty_corr,
            'mean_uncertainty': np.mean(uncertainties),
            'uncertainty_std': np.std(uncertainties)
        }

    def _evaluate_path_lengths(self,
                               predicted_paths: List[List[PathStep]],
                               actual_paths: List[List[PathStep]]) -> float:
        """경로 길이 평가"""
        pred_lengths = [len(path) for path in predicted_paths]
        true_lengths = [len(path) for path in actual_paths]

        return np.mean(np.abs(
            np.array(pred_lengths) - np.array(true_lengths)
        ))

    def _evaluate_structure_similarity(self,
                                       predicted_paths: List[List[PathStep]],
                                       actual_paths: List[List[PathStep]]) -> float:
        """구조 유사도 평가"""
        similarities = []

        for pred_path, true_path in zip(predicted_paths, actual_paths):
            path_similarities = []
            for pred_step, true_step in zip(pred_path, true_path):
                similarity = self._calculate_structure_similarity(
                    pred_step.final_structure,
                    true_step.final_structure
                )
                path_similarities.append(similarity)

            similarities.append(np.mean(path_similarities))

        return np.mean(similarities)

    def _evaluate_energy_profiles(self,
                                  predicted_paths: List[List[PathStep]],
                                  actual_paths: List[List[PathStep]]) -> float:
        """에너지 프로파일 평가"""
        profile_errors = []

        for pred_path, true_path in zip(predicted_paths, actual_paths):
            pred_energies = [step.energy_final for step in pred_path]
            true_energies = [step.energy_final for step in true_path]

            # 프로파일 길이 맞추기
            min_len = min(len(pred_energies), len(true_energies))
            pred_energies = pred_energies[:min_len]
            true_energies = true_energies[:min_len]

            # RMSE 계산
            error = np.sqrt(mean_squared_error(true_energies, pred_energies))
            profile_errors.append(error)

        return np.mean(profile_errors)

    def _evaluate_path_continuity(self,
                                  paths: List[List[PathStep]]) -> float:
        """경로 연속성 평가"""
        continuity_scores = []

        for path in paths:
            step_scores = []
            for i in range(len(path) - 1):
                score = self._calculate_step_continuity(
                    path[i].final_structure,
                    path[i + 1].initial_structure
                )
                step_scores.append(score)

            continuity_scores.append(np.mean(step_scores))

        return np.mean(continuity_scores)

    def _calculate_structure_similarity(self,
                                        struct1: Structure,
                                        struct2: Structure) -> float:
        """구조 유사도 계산"""
        # RMSD 기본 계산
        rmsd_score = self._calculate_rmsd(struct1, struct2)

        # 그래프 기반 유사도
        graph1 = self._structure_to_graph(struct1)
        graph2 = self._structure_to_graph(struct2)

        # 노드 특성 유사도 (Cosine Similarity)
        node_sim = F.cosine_similarity(
            graph1.x.mean(dim=0, keepdim=True),
            graph2.x.mean(dim=0, keepdim=True)
        ).item()

        # 엣지 분포 유사도 (Wasserstein Distance)
        edge_dist1 = graph1.edge_attr.view(-1).numpy()
        edge_dist2 = graph2.edge_attr.view(-1).numpy()
        edge_sim = 1.0 / (1.0 + wasserstein_distance(edge_dist1, edge_dist2))

        # 위상 유사도 (Spectral Distance)
        spec_sim = self._calculate_spectral_similarity(graph1, graph2)

        # 종합 점수 계산 (가중 평균)
        weights = {
            'rmsd': 0.4,
            'node': 0.2,
            'edge': 0.2,
            'spectral': 0.2
        }

        similarity = (
                weights['rmsd'] * (1.0 / (1.0 + rmsd_score)) +
                weights['node'] * node_sim +
                weights['edge'] * edge_sim +
                weights['spectral'] * spec_sim
        )

        return similarity

    def _calculate_step_continuity(self,
                                   struct1: Structure,
                                   struct2: Structure) -> float:
        """단계 연속성 계산"""
        graph1 = self._structure_to_graph(struct1)
        graph2 = self._structure_to_graph(struct2)

        # 노드 특성 연속성
        node_smoothness = self._calculate_feature_smoothness(
            graph1.x,
            graph2.x
        )

        # 엣지 연속성
        edge_consistency = self._calculate_edge_consistency(
            graph1.edge_index, graph1.edge_attr,
            graph2.edge_index, graph2.edge_attr
        )

        # 위상 연속성
        topo_persistence = self._calculate_topological_persistence(
            graph1, graph2
        )

        # 구조 변화량
        structural_change = self._calculate_structural_change(
            struct1, struct2
        )

        # 종합 연속성 점수
        weights = {
            'node': 0.3,
            'edge': 0.3,
            'topo': 0.2,
            'struct': 0.2
        }

        continuity = (
                weights['node'] * node_smoothness +
                weights['edge'] * edge_consistency +
                weights['topo'] * topo_persistence +
                weights['struct'] * (1.0 - structural_change)
        )

        return continuity

    def _calculate_feature_smoothness(self,
                                      features1: torch.Tensor,
                                      features2: torch.Tensor) -> float:
        """노드 특성 연속성 계산"""
        # L2 norm of feature differences
        diff = features2 - features1
        smoothness = 1.0 - torch.norm(diff, p=2).item() / (features1.size(0) * features1.size(1))
        return max(0.0, smoothness)

    def _calculate_edge_consistency(self,
                                    edge_index1: torch.Tensor,
                                    edge_attr1: torch.Tensor,
                                    edge_index2: torch.Tensor,
                                    edge_attr2: torch.Tensor) -> float:
        """엣지 연속성 계산"""
        # 엣지 집합 비교
        edges1 = set(map(tuple, edge_index1.t().tolist()))
        edges2 = set(map(tuple, edge_index2.t().tolist()))

        # Jaccard similarity for edge sets
        intersection = len(edges1.intersection(edges2))
        union = len(edges1.union(edges2))

        if union == 0:
            return 0.0

        return intersection / union

    def _calculate_topological_persistence(self,
                                           graph1: Data,
                                           graph2: Data) -> float:
        """위상 연속성 계산"""
        # 그래프 라플라시안 eigen value 비교
        L1 = self._get_graph_laplacian(graph1)
        L2 = self._get_graph_laplacian(graph2)

        eig1 = torch.linalg.eigvalsh(L1)[:10]  # 처음 10개 eigenvalue만 사용
        eig2 = torch.linalg.eigvalsh(L2)[:10]

        # Spectral distance
        diff = torch.norm(eig1 - eig2, p=2)
        persistence = 1.0 / (1.0 + diff.item())

        return persistence