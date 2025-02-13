import argparse
import asyncio
from pathlib import Path
import logging
from typing import Optional
import sys

from core.system import MLDFTSystem
from config.settings import load_config
from utils.logger import get_logger


async def train_models(
        config_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        model_type: str = "all",
        resume: bool = False,
        verbose: bool = False
):
    """ML 모델 학습"""
    # 로거 설정
    logger = get_logger(
        "train",
        level=logging.DEBUG if verbose else logging.INFO
    )

    try:
        # 설정 로드
        logger.info("Loading configuration...")
        config = load_config(config_path)

        # 시스템 초기화
        logger.info("Initializing ML-DFT system...")
        system = MLDFTSystem(config)

        # 데이터 디렉토리 설정
        data_dir = data_dir or Path("data")
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # 출력 디렉토리 설정
        output_dir = output_dir or Path("models")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 학습 데이터 로드
        logger.info("Loading training data...")
        training_data = await system.data_manager.load_training_data(data_dir)

        # 모델 선택
        if model_type == "all":
            models_to_train = ["property", "path", "stability"]
        else:
            models_to_train = [model_type]

        # 모델 학습
        for model_type in models_to_train:
            logger.info(f"Training {model_type} model...")

            # 체크포인트 복원 (필요한 경우)
            if resume:
                checkpoint_path = output_dir / f"{model_type}_checkpoint.pt"
                if checkpoint_path.exists():
                    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                    await system.ml_manager.load_checkpoint(checkpoint_path)

            # 모델 학습
            try:
                if model_type == "property":
                    await system.ml_manager.train_property_model(
                        training_data,
                        output_dir=output_dir
                    )
                elif model_type == "path":
                    await system.ml_manager.train_path_model(
                        training_data,
                        output_dir=output_dir
                    )
                elif model_type == "stability":
                    await system.ml_manager.train_stability_model(
                        training_data,
                        output_dir=output_dir
                    )

                logger.info(f"Successfully trained {model_type} model")

            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                if verbose:
                    logger.exception(e)
                continue

            # 모델 평가
            logger.info(f"Evaluating {model_type} model...")
            metrics = await system.ml_manager.evaluate_model(
                model_type,
                training_data
            )

            # 메트릭스 로깅
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if verbose:
            logger.exception(e)
        sys.exit(1)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Train ML-DFT models")

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to data directory"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to output directory"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["all", "property", "path", "stability"],
        default="all",
        help="Type of model to train"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # 비동기 실행
    asyncio.run(
        train_models(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_type=args.model_type,
            resume=args.resume,
            verbose=args.verbose
        )
    )


if __name__ == "__main__":
    main()