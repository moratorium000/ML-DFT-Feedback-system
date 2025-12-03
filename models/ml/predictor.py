from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

from core.interfaces import (
    Structure,
    PredictionResult,
    PathStep
)
from core.protocols import IModelPredictor
from utils.constants import (
    ELECTRONEGATIVITY,
    ATOMIC_RADIUS,
    IONIZATION_ENERGY,
    MAX_BOND_LENGTH,
    ATOMIC_NUMBERS,
    ELEMENT_SYMBOLS
)


class StructureEncoder(nn.Module):
    """원자 구조 인코더"""

    def __init__(self,
                 node_features: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return global_mean_pool(x, batch)


class PropertyPredictor(nn.Module):
    """물성 예측 모델"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 n_properties: int = 5):
        super().__init__()
        self.encoder = StructureEncoder(input_dim, hidden_dim)

        self.property_heads = nn.ModuleDict({
            'energy': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ),
            'forces': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)
            ),
            'band_gap': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        })

        self.uncertainty_heads = nn.ModuleDict({
            prop: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            ) for prop in self.property_heads.keys()
        })


class PathPredictor(nn.Module):
    """경로 예측 모델"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.encoder = StructureEncoder(input_dim, hidden_dim)
        self.path_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )


class MLPredictor(IModelPredictor):
    """ML 예측 시스템"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.property_model = PropertyPredictor(input_dim, hidden_dim).to(device)
        self.path_model = PathPredictor(input_dim, hidden_dim).to(device)

        self.property_criterion = nn.MSELoss()
        self.path_criterion = nn.MSELoss()

        # 물성 범위 (정규화 및 신뢰도 계산용)
        self.property_ranges = {
            'energy': (-10.0, 0.0),       # eV/atom
            'band_gap': (0.0, 10.0),      # eV
            'forces': (-5.0, 5.0),        # eV/Å
            'formation_energy': (-5.0, 2.0)  # eV/atom
        }

        # 경로 수렴 임계값
        self.convergence_threshold = 0.1
        self.max_path_steps = 50

    async def predict_properties(self,
                                 structure: Structure) -> PredictionResult:
        """물성 예측"""
        self.property_model.eval()
        with torch.no_grad():
            # 구조를 그래프로 변환
            graph = self._structure_to_graph(structure)
            graph = graph.to(self.device)

            # 인코딩
            encoded = self.property_model.encoder(
                graph.x,
                graph.edge_index,
                graph.batch
            )

            # 물성 예측
            predictions = {}
            uncertainties = {}

            for prop_name, head in self.property_model.property_heads.items():
                pred = head(encoded)
                uncert = self.property_model.uncertainty_heads[prop_name](encoded)

                predictions[prop_name] = pred.cpu().numpy()
                uncertainties[prop_name] = uncert.cpu().numpy()

            # 신뢰도 점수 계산
            confidence = self._calculate_confidence(uncertainties)

            return PredictionResult(
                predicted_values=predictions,
                uncertainty=uncertainties,
                confidence_score=confidence,
                prediction_details={}
            )

    async def predict_path(self,
                           initial: Structure,
                           target: Structure) -> List[PathStep]:
        """경로 예측"""
        self.path_model.eval()
        with torch.no_grad():
            # 구조를 그래프로 변환
            initial_graph = self._structure_to_graph(initial).to(self.device)
            target_graph = self._structure_to_graph(target).to(self.device)

            # batch 인덱스 추가
            initial_graph.batch = torch.zeros(initial_graph.x.size(0), dtype=torch.long, device=self.device)
            target_graph.batch = torch.zeros(target_graph.x.size(0), dtype=torch.long, device=self.device)

            # 초기/목표 구조 인코딩
            initial_encoded = self.path_model.encoder(
                initial_graph.x,
                initial_graph.edge_index,
                initial_graph.batch
            )
            target_encoded = self.path_model.encoder(
                target_graph.x,
                target_graph.edge_index,
                target_graph.batch
            )

            # 경로 생성
            path = []
            current = initial_encoded
            current_structure = initial
            step_count = 0

            while not self._is_target_reached(current, target_encoded) and step_count < self.max_path_steps:
                # 다음 구조 예측
                transition = self.path_model.transition_predictor(
                    torch.cat([current, target_encoded], dim=1)
                )

                # 구조로 변환
                next_structure = self._decode_structure(transition)
                confidence = self._calculate_step_confidence(transition, current, target_encoded)

                # 경로에 추가
                path.append(PathStep(
                    step_type='predicted',
                    initial_structure=current_structure,
                    final_structure=next_structure,
                    energy_initial=0.0,
                    energy_final=0.0,
                    energy_barrier=None,
                    transformation_matrix=None,
                    atomic_mapping=None,
                    dft_results=None,
                    ml_predictions=None,
                    success=True,
                    reversible=True,
                    confidence=confidence
                ))

                # 다음 스텝 준비
                next_graph = self._structure_to_graph(next_structure).to(self.device)
                next_graph.batch = torch.zeros(next_graph.x.size(0), dtype=torch.long, device=self.device)
                current = self.path_model.encoder(
                    next_graph.x,
                    next_graph.edge_index,
                    next_graph.batch
                )
                current_structure = next_structure
                step_count += 1

            return path

    def _is_target_reached(self, current: torch.Tensor, target: torch.Tensor) -> bool:
        """목표 구조에 도달했는지 확인"""
        distance = torch.norm(current - target).item()
        return distance < self.convergence_threshold

    def _calculate_step_confidence(self,
                                   transition: torch.Tensor,
                                   current: torch.Tensor,
                                   target: torch.Tensor) -> float:
        """스텝 신뢰도 계산"""
        # 목표 방향으로의 진행 정도 계산
        current_dist = torch.norm(current - target).item()
        transition_magnitude = torch.norm(transition).item()

        # 신뢰도: 전이 크기가 적절하고 목표로 향할수록 높음
        if current_dist == 0:
            return 1.0

        # 정규화된 신뢰도 (0-1 범위)
        confidence = min(1.0, 1.0 / (1.0 + transition_magnitude / current_dist))
        return float(confidence)

    def _get_formula(self, atomic_numbers: List[int]) -> str:
        """원자 번호 리스트에서 화학식 생성"""
        from collections import Counter
        counts = Counter(atomic_numbers)
        formula_parts = []
        for z, count in sorted(counts.items()):
            symbol = ELEMENT_SYMBOLS.get(z, f'X{z}')
            if count == 1:
                formula_parts.append(symbol)
            else:
                formula_parts.append(f"{symbol}{count}")
        return ''.join(formula_parts)

    async def estimate_uncertainty(self,
                                   prediction: PredictionResult) -> Dict[str, float]:
        """불확실성 추정"""
        return {
            prop: float(uncert.mean())
            for prop, uncert in prediction.uncertainty.items()
        }

    def _structure_to_graph(self, structure: Structure) -> Data:
        """구조를 그래프로 변환"""
        # 원자 특성 벡터 생성
        num_atoms = len(structure.atomic_numbers)
        node_features = []
        for z in structure.atomic_numbers:
            z_int = int(z)
            # 원자 특성: [원자 번호, 전기음성도, 원자 반지름, 이온화 에너지]
            features = [
                float(z_int),
                ELECTRONEGATIVITY.get(z_int, 0.0),
                ATOMIC_RADIUS.get(z_int, 1.0),
                IONIZATION_ENERGY.get(z_int, 10.0)
            ]
            node_features.append(features)

        # 엣지 생성 (거리 기반 연결)
        edge_index = []
        edge_attr = []
        positions = np.array(structure.positions)
        lattice = np.array(structure.lattice_vectors)

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    # 최소 이미지 규약 적용
                    diff = positions[j] - positions[i]
                    diff = diff - np.round(diff)
                    cart_diff = np.dot(diff, lattice)
                    distance = float(np.linalg.norm(cart_diff))

                    if distance <= MAX_BOND_LENGTH:
                        edge_index.append([i, j])
                        edge_attr.append([distance])

        # 엣지가 없으면 자기 연결 추가
        if not edge_index:
            edge_index = [[i, i] for i in range(num_atoms)]
            edge_attr = [[0.0] for _ in range(num_atoms)]

        # PyTorch Geometric Data 객체 생성
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            pos=torch.tensor(positions, dtype=torch.float),
            cell=torch.tensor(lattice, dtype=torch.float)
        )

    def _decode_structure(self, encoded: torch.Tensor) -> Structure:
        """인코딩된 표현을 구조로 변환"""
        # 텐서를 numpy 배열로 변환
        decoded = encoded.detach().cpu().numpy()

        # 구조 파라미터 추출 (1D 벡터를 구조로 변환)
        if decoded.ndim == 1:
            decoded = decoded.reshape(1, -1)

        # 유효 원자 번호 목록
        valid_atomic_numbers = list(ELEMENT_SYMBOLS.keys())

        # 특성 차원에서 원자 수 추정 (4개 특성/원자)
        feature_dim = decoded.shape[-1]
        n_atoms = max(1, feature_dim // 4)

        # 원자 번호 예측 (가장 가까운 실제 원자 번호로 매핑)
        atomic_numbers = []
        for i in range(n_atoms):
            if i * 4 < feature_dim:
                z_pred = abs(decoded[0, i * 4]) if decoded.ndim == 2 else abs(decoded[i * 4])
                # 가장 가까운 실제 원자 번호 찾기
                z = min(valid_atomic_numbers, key=lambda x: abs(x - z_pred))
                atomic_numbers.append(z)
            else:
                atomic_numbers.append(14)  # Si 기본값

        # 위치 좌표 생성 (랜덤 초기화)
        positions = np.random.rand(n_atoms, 3)

        # 격자 벡터는 별도로 처리 필요 (여기서는 원본 유지 가정)
        lattice_vectors = np.eye(3) * 10.0  # 기본값으로 10Å 큐빅 셀

        return Structure(
            atomic_numbers=np.array(atomic_numbers),
            positions=positions,
            lattice_vectors=lattice_vectors,
            cell_params={"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            formula=self._get_formula(atomic_numbers)
        )

    def _calculate_confidence(self, uncertainties: Dict[str, np.ndarray]) -> float:
        """신뢰도 점수 계산"""
        # 각 물성의 상대 불확실성 계산
        relative_uncertainties = []

        for prop_name, uncertainty in uncertainties.items():
            if prop_name in self.property_ranges:
                # 물성의 예상 범위로 정규화
                prop_range = self.property_ranges[prop_name]
                range_size = prop_range[1] - prop_range[0]
                relative_uncertainty = np.mean(uncertainty) / range_size
                relative_uncertainties.append(relative_uncertainty)

        if not relative_uncertainties:
            return 0.0

        # 전체 불확실성의 평균 계산
        mean_uncertainty = np.mean(relative_uncertainties)

        # 신뢰도 점수 계산 (0-1 범위로 변환)
        confidence = 1.0 - min(mean_uncertainty, 1.0)

        # 신뢰도 점수를 시그모이드 함수로 조정하여 극단값 방지
        confidence = 1.0 / (1.0 + np.exp(-5 * (confidence - 0.5)))

        return float(confidence)