from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from predictor import PropertyPredictor, PathPredictor
from core.interfaces import Structure, DFTResult, MLModelState
from utils.logger import get_logger


class ModelTrainer:
    """ML 모델 학습기"""

    def __init__(self,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = config
        self.device = device
        self.logger = get_logger(__name__)

        # 모델 초기화
        self.property_model = PropertyPredictor(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim']
        ).to(device)

        self.path_model = PathPredictor(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim']
        ).to(device)

        # 옵티마이저 설정
        self.property_optimizer = torch.optim.Adam(
            self.property_model.parameters(),
            lr=config['learning_rate']
        )

        self.path_optimizer = torch.optim.Adam(
            self.path_model.parameters(),
            lr=config['learning_rate']
        )

        # 학습 이력
        self.training_history = {
            'property_loss': [],
            'path_loss': [],
            'validation_metrics': []
        }

    async def train_property_model(self,
                                   train_loader: DataLoader,
                                   val_loader: Optional[DataLoader] = None,
                                   n_epochs: int = 100) -> MLModelState:
        """물성 예측 모델 학습"""
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # 학습
            train_loss = await self._train_property_epoch(train_loader)
            self.training_history['property_loss'].append(train_loss)

            # 검증
            if val_loader is not None:
                val_loss, metrics = await self._validate_property_model(val_loader)
                self.training_history['validation_metrics'].append(metrics)

                # Early stopping 체크
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint('property_model_best.pt', self.property_model)
                else:
                    patience_counter += 1

                if patience_counter >= self.config['patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            self.logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}"
            )

        return self._create_model_state('property')

    async def train_path_model(self,
                               train_loader: DataLoader,
                               val_loader: Optional[DataLoader] = None,
                               n_epochs: int = 100) -> MLModelState:
        """경로 예측 모델 학습"""
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # 학습
            train_loss = await self._train_path_epoch(train_loader)
            self.training_history['path_loss'].append(train_loss)

            # 검증
            if val_loader is not None:
                val_loss, metrics = await self._validate_path_model(val_loader)

                # Early stopping 체크
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint('path_model_best.pt', self.path_model)
                else:
                    patience_counter += 1

                if patience_counter >= self.config['patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            self.logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}"
            )

        return self._create_model_state('path')

    async def _train_property_epoch(self,
                                    train_loader: DataLoader) -> float:
        """물성 모델 한 에포크 학습"""
        self.property_model.train()
        total_loss = 0

        for batch in train_loader:
            self.property_optimizer.zero_grad()

            # 배치를 GPU로 이동
            batch = batch.to(self.device)

            # 예측
            encoded = self.property_model.encoder(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            # 각 물성에 대한 손실 계산
            loss = 0
            for prop_name, head in self.property_model.property_heads.items():
                pred = head(encoded)
                target = batch[f"{prop_name}_target"]
                prop_loss = F.mse_loss(pred, target)
                loss += prop_loss

                # 불확실성 손실 추가
                uncert = self.property_model.uncertainty_heads[prop_name](encoded)
                uncertainty_loss = self._uncertainty_loss(pred, target, uncert)
                loss += uncertainty_loss

            loss.backward()
            self.property_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    async def _train_path_epoch(self,
                                train_loader: DataLoader) -> float:
        """경로 모델 한 에포크 학습"""
        self.path_model.train()
        total_loss = 0

        for batch in train_loader:
            self.path_optimizer.zero_grad()

            # 배치를 GPU로 이동
            batch = batch.to(self.device)

            # 시작/목표 구조 인코딩
            initial_encoded = self.path_model.encoder(
                batch.initial_x,
                batch.initial_edge_index,
                batch.initial_batch
            )

            target_encoded = self.path_model.encoder(
                batch.target_x,
                batch.target_edge_index,
                batch.target_batch
            )

            # 경로 예측
            path_pred = self.path_model.transition_predictor(
                torch.cat([initial_encoded, target_encoded], dim=1)
            )

            # 손실 계산
            loss = F.mse_loss(path_pred, batch.path_target)

            loss.backward()
            self.path_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _uncertainty_loss(self,
                          pred: torch.Tensor,
                          target: torch.Tensor,
                          uncertainty: torch.Tensor) -> torch.Tensor:
        """불확실성을 고려한 손실 함수"""
        return torch.mean(
            0.5 * torch.exp(-uncertainty) * (pred - target) ** 2 +
            0.5 * uncertainty
        )

    def _save_checkpoint(self,
                         filename: str,
                         model: nn.Module):
        """모델 체크포인트 저장"""
        save_path = Path(self.config['checkpoint_dir']) / filename
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, save_path)

    def _create_model_state(self, model_type: str) -> MLModelState:
        """모델 상태 생성"""
        return MLModelState(
            model_type=model_type,
            training_iterations=len(self.training_history[f'{model_type}_loss']),
            performance_metrics=self._get_latest_metrics(),
            last_update=datetime.now(),
            model_parameters=self.config,
            training_history=self.training_history
        )