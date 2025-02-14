from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from core.interfaces import (
    Structure,
    PredictionResult,
    PathStep
)
from core.protocols import IModelPredictor


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

        self.scaler = StandardScaler()
        self.property_criterion = nn.MSELoss()
        self.path_criterion = nn.MSELoss()

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

            while not self._is_target_reached(current, target_encoded):
                # 다음 구조 예측
                transition = self.path_model.transition_predictor(
                    torch.cat([current, target_encoded], dim=1)
                )

                # 구조로 변환
                next_structure = self._decode_structure(transition)

                # 경로에 추가
                path.append(PathStep(
                    initial_structure=self._decode_structure(current),
                    final_structure=next_structure,
                    confidence=self._calculate_step_confidence(transition)
                ))

                current = self.path_model.encoder(
                    self._structure_to_graph(next_structure).to(self.device)
                )

            return path

    async def estimate_uncertainty(self,
                                   prediction: PredictionResult) -> Dict[str, float]:
        """불확실성 추정"""
        return {
            prop: float(uncert.mean())
            for prop, uncert in prediction.uncertainty.items()
        }

    def _structure_to_graph(self, structure: Structure):
        """구조를 그래프로 변환"""
        import torch
        from torch_geometric.data import Data

        # 원자 특성 벡터 생성
        num_atoms = len(structure.atomic_numbers)
        node_features = []
        for z in structure.atomic_numbers:
            # 원자 특성: [원자 번호, 전기음성도, 원자 반지름, 이온화 에너지]
            features = [
                z,
                ELECTRONEGATIVITY.get(z, 0.0),
                ATOMIC_RADIUS.get(z, 0.0),
                IONIZATION_ENERGY.get(z, 0.0)
            ]
            node_features.append(features)

        # 엣지 생성 (거리 기반 연결)
        edge_index = []
        edge_attr = []
        positions = structure.positions
        lattice = structure.lattice_vectors

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    # 최소 이미지 규약 적용
                    diff = positions[j] - positions[i]
                    diff = diff - np.round(diff)
                    cart_diff = np.dot(diff, lattice)
                    distance = np.linalg.norm(cart_diff)

                    if distance <= MAX_BOND_LENGTH:
                        edge_index.append([i, j])
                        edge_attr.append([distance])

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

        # 구조 파라미터 추출
        batch_size, feature_dim = decoded.shape
        n_atoms = batch_size // 4  # 각 원자당 4개의 특성

        # 원자 특성 복원
        atomic_features = decoded.reshape(n_atoms, 4)

        # 원자 번호 예측 (가장 가까운 실제 원자 번호로 매핑)
        atomic_numbers = []
        for features in atomic_features:
            z_pred = features[0]  # 첫 번째 특성이 원자 번호
            # 가장 가까운 실제 원자 번호 찾기
            z = min(ATOMIC_NUMBERS, key=lambda x: abs(x - z_pred))
            atomic_numbers.append(z)

        # 위치 좌표 생성
        positions = decoded[:, 1:4]  # 나머지 3개 특성을 위치 좌표로 사용

        # 격자 벡터는 별도로 처리 필요 (여기서는 원본 유지 가정)
        lattice_vectors = np.eye(3) * 10.0  # 기본값으로 10Å 큐빅 셀

        return Structure(
            atomic_numbers=np.array(atomic_numbers),
            positions=positions,
            lattice_vectors=lattice_vectors,
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