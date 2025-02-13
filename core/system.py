from typing import Dict, List, Optional, Union
from pathlib import Path
import asyncio
import logging

from manager import (
    PrototypeManager,
    DFTManager,
    MLManager,
    PathManager
)
from interfaces import (
    Structure,
    MutationResult,
    DFTResult,
    PathStep,
    OptimizationResult
)
from data.cache import CacheManager
from data.database import DatabaseManager
from utils.logger import setup_logger
from config.settings import SystemConfig


class MLDFTSystem:
    """ML-DFT 피드백 시스템의 메인 클래스"""

    def __init__(self, config: SystemConfig):
        """
        시스템 초기화

        Args:
            config: 시스템 설정
        """
        self.config = config
        self.logger = setup_logger(__name__)

        # 핵심 매니저 초기화
        self.prototype_manager = PrototypeManager(config.prototype)
        self.dft_manager = DFTManager(config.dft)
        self.ml_manager = MLManager(config.ml)
        self.path_manager = PathManager(config.path)

        # 데이터 관리 시스템 초기화
        self.cache_manager = CacheManager(config.cache)
        self.db_manager = DatabaseManager(config.database)

        self.logger.info("MLDFTSystem initialized successfully")

    async def optimize_structure(self,
                                 initial_structure: Structure,
                                 target_properties: Dict[str, float],
                                 accepted_error: Dict[str, float]) -> OptimizationResult:
        """
        구조 최적화 실행

        Args:
            initial_structure: 시작 구조
            target_properties: 목표 물성값
            accepted_error: 허용 오차 범위

        Returns:
            최적화 결과
        """
        try:
            # 초기 구조 등록
            prototype_id = await self.prototype_manager.register_prototype(
                initial_structure,
                {"target_properties": target_properties}
            )

            optimization_history = []
            current_step = 0

            while not self._convergence_reached(optimization_history):
                if current_step >= self.config.max_iterations:
                    self.logger.warning("Maximum iterations reached")
                    break

                # ML 예측 및 경로 생성
                predicted_paths = await self.ml_manager.predict_paths(
                    initial_structure,
                    target_properties
                )

                # 경로 평가 및 선택
                selected_path = await self.path_manager.evaluate_and_select_path(
                    predicted_paths,
                    target_properties
                )

                # DFT 검증
                dft_results = await self.dft_manager.validate_path(selected_path)

                # 결과 분석 및 피드백
                feedback = await self._analyze_results(
                    dft_results,
                    target_properties,
                    accepted_error
                )

                # 데이터베이스 업데이트
                await self._update_database(feedback)

                # ML 모델 업데이트
                await self.ml_manager.update_models(dft_results)

                # 히스토리 업데이트
                optimization_history.append(feedback)
                current_step += 1

                self.logger.info(f"Completed optimization step {current_step}")

            return self._prepare_optimization_result(optimization_history)

        except Exception as e:
            self.logger.error(f"Error during structure optimization: {e}")
            raise

    async def _analyze_results(self,
                               dft_results: List[DFTResult],
                               target_properties: Dict[str, float],
                               accepted_error: Dict[str, float]) -> Dict:
        """결과 분석 및 피드백 생성"""
        property_matches = {}
        deviations = {}

        for prop, target in target_properties.items():
            actual = self._extract_property(dft_results, prop)
            error_range = accepted_error[prop]

            match_score = self._calculate_match_score(actual, target, error_range)
            deviation = actual - target

            property_matches[prop] = match_score
            deviations[prop] = deviation

        return {
            "property_matches": property_matches,
            "deviations": deviations,
            "dft_results": dft_results
        }

    def _convergence_reached(self, history: List[Dict]) -> bool:
        """수렴 여부 확인"""
        if not history:
            return False

        recent_results = history[-self.config.convergence_window:]

        # 물성 매칭 점수 확인
        match_scores = [
            result["property_matches"]
            for result in recent_results
        ]

        # 모든 물성이 허용 오차 내에 있는지 확인
        for scores in match_scores:
            if any(score < self.config.convergence_threshold
                   for score in scores.values()):
                return False

        # 변화량이 충분히 작은지 확인
        if len(recent_results) >= 2:
            changes = self._calculate_changes(recent_results)
            if any(change > self.config.min_change_threshold
                   for change in changes):
                return False

        return True

    async def _update_database(self, feedback: Dict):
        """데이터베이스 업데이트"""
        try:
            # DFT 결과 저장
            await self.db_manager.store_dft_results(feedback["dft_results"])

            # 물성 매칭 결과 저장
            await self.db_manager.store_property_matches(
                feedback["property_matches"]
            )

            # 최적화 히스토리 업데이트
            await self.db_manager.update_optimization_history(feedback)

        except Exception as e:
            self.logger.error(f"Database update failed: {e}")
            raise

    def _prepare_optimization_result(self,
                                     history: List[Dict]) -> OptimizationResult:
        """최적화 결과 준비"""
        best_result = max(history,
                          key=lambda x: sum(x["property_matches"].values()))

        return OptimizationResult(
            final_structure=best_result["dft_results"][-1].final_structure,
            property_matches=best_result["property_matches"],
            optimization_path=history,
            convergence_achieved=self._convergence_reached(history)
        )