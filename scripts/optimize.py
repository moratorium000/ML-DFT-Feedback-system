import argparse
import asyncio
from pathlib import Path
import logging
from typing import Optional, Dict
import sys
import json
import yaml

from core.system import MLDFTSystem
from config.settings import load_config
from utils.logger import get_logger
from core.interfaces import Structure


async def optimize_structure(
        input_structure: Path,
        target_properties: Dict,
        config_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        max_iterations: int = 100,
        tolerance: float = 0.01,
        verbose: bool = False
):
    """구조 최적화 실행"""
    # 로거 설정
    logger = get_logger(
        "optimize",
        level=logging.DEBUG if verbose else logging.INFO
    )

    try:
        # 설정 로드
        logger.info("Loading configuration...")
        config = load_config(config_path)

        # 시스템 초기화
        logger.info("Initializing ML-DFT system...")
        system = MLDFTSystem(config)

        # 출력 디렉토리 설정
        output_dir = output_dir or Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 입력 구조 로드
        logger.info(f"Loading input structure from {input_structure}")
        structure = system.prototype_manager.load_structure(input_structure)

        # 최적화 실행
        logger.info("Starting optimization...")
        optimization_result = await system.optimize_structure(
            structure=structure,
            target_properties=target_properties,
            max_iterations=max_iterations,
            tolerance=tolerance
        )

        # 결과 분석
        if optimization_result.convergence_achieved:
            logger.info("Optimization converged successfully!")
        else:
            logger.warning("Optimization did not fully converge")

        # 결과 저장
        logger.info("Saving results...")

        # 최종 구조 저장
        final_structure_path = output_dir / "final_structure.json"
        system.prototype_manager.save_structure(
            optimization_result.final_structure,
            final_structure_path
        )

        # 최적화 경로 저장
        path_data = {
            'iterations': optimization_result.total_iterations,
            'computation_time': optimization_result.computation_time,
            'convergence_achieved': optimization_result.convergence_achieved,
            'property_matches': optimization_result.property_matches,
            'history': [
                {
                    'iteration': i,
                    'property_values': step.property_values,
                    'improvements': step.improvements
                }
                for i, step in enumerate(optimization_result.history)
            ]
        }

        path_file = output_dir / "optimization_path.json"
        with open(path_file, 'w') as f:
            json.dump(path_data, f, indent=2)

        # 성능 메트릭스 저장
        metrics = {
            'final_score': optimization_result.final_score,
            'property_matches': optimization_result.property_matches,
            'computation_time': optimization_result.computation_time,
            'total_iterations': optimization_result.total_iterations,
            'convergence_achieved': optimization_result.convergence_achieved
        }

        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # 결과 요약 출력
        logger.info("\nOptimization Summary:")
        logger.info(f"Total Iterations: {optimization_result.total_iterations}")
        logger.info(f"Computation Time: {optimization_result.computation_time:.2f}s")
        logger.info(f"Final Score: {optimization_result.final_score:.4f}")
        logger.info("\nProperty Matches:")
        for prop, value in optimization_result.property_matches.items():
            logger.info(f"{prop}: {value:.4f}")

        logger.info(f"\nResults saved to {output_dir}")

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if verbose:
            logger.exception(e)
        sys.exit(1)


def load_target_properties(target_file: Path) -> Dict:
    """목표 물성 로드"""
    with open(target_file) as f:
        if target_file.suffix == '.yaml':
            return yaml.safe_load(f)
        elif target_file.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {target_file.suffix}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Optimize structure")

    parser.add_argument(
        "input_structure",
        type=Path,
        help="Path to input structure file"
    )

    parser.add_argument(
        "target_properties",
        type=Path,
        help="Path to target properties file (JSON/YAML)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to output directory"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of optimization iterations"
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Convergence tolerance"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # 목표 물성 로드
    target_properties = load_target_properties(args.target_properties)

    # 비동기 실행
    asyncio.run(
        optimize_structure(
            input_structure=args.input_structure,
            target_properties=target_properties,
            config_path=args.config,
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            verbose=args.verbose
        )
    )


if __name__ == "__main__":
    main()