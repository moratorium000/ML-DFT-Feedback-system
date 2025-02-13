import argparse
import asyncio
from pathlib import Path
import logging
from typing import Optional, List, Dict
import sys
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from core.system import MLDFTSystem
from config.settings import load_config
from utils.logger import get_logger


async def analyze_results(
        results_dir: Path,
        output_dir: Optional[Path] = None,
        analysis_type: str = "all",
        config_path: Optional[Path] = None,
        verbose: bool = False
):
    """결과 분석 실행"""
    # 로거 설정
    logger = get_logger(
        "analyze",
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
        output_dir = output_dir or results_dir / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 결과 로드
        logger.info("Loading optimization results...")
        results = await load_results(results_dir)

        # 분석 수행
        analyses = []
        if analysis_type in ["all", "convergence"]:
            analyses.append(analyze_convergence(results, output_dir))
        if analysis_type in ["all", "property"]:
            analyses.append(analyze_properties(results, output_dir))
        if analysis_type in ["all", "structure"]:
            analyses.append(analyze_structures(results, output_dir))
        if analysis_type in ["all", "performance"]:
            analyses.append(analyze_performance(results, output_dir))

        # 분석 실행
        analysis_results = await asyncio.gather(*analyses)

        # 종합 보고서 생성
        logger.info("Generating comprehensive report...")
        await generate_report(analysis_results, output_dir)

        logger.info(f"Analysis completed. Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if verbose:
            logger.exception(e)
        sys.exit(1)


async def load_results(results_dir: Path) -> Dict:
    """결과 데이터 로드"""
    final_structure = json.loads(
        (results_dir / "final_structure.json").read_text()
    )
    optimization_path = json.loads(
        (results_dir / "optimization_path.json").read_text()
    )
    metrics = json.loads(
        (results_dir / "metrics.json").read_text()
    )

    return {
        'final_structure': final_structure,
        'optimization_path': optimization_path,
        'metrics': metrics
    }


async def analyze_convergence(results: Dict, output_dir: Path) -> Dict:
    """수렴성 분석"""
    history = results['optimization_path']['history']

    # 데이터 준비
    iterations = range(len(history))
    improvements = [step['improvements'] for step in history]
    property_values = [step['property_values'] for step in history]

    # 수렴 그래프 생성
    plt.figure(figsize=(10, 6))
    for prop in property_values[0].keys():
        values = [step['property_values'][prop] for step in history]
        plt.plot(iterations, values, label=prop)

    plt.xlabel('Iteration')
    plt.ylabel('Property Value')
    plt.title('Convergence of Properties')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'convergence.png')
    plt.close()

    return {
        'type': 'convergence',
        'data': {
            'total_iterations': len(history),
            'final_improvements': improvements[-1],
            'convergence_rate': calculate_convergence_rate(improvements)
        }
    }


async def analyze_properties(results: Dict, output_dir: Path) -> Dict:
    """물성 분석"""
    final_properties = results['metrics']['property_matches']
    target_properties = results['optimization_path']['target_properties']

    # 물성 비교 그래프
    plt.figure(figsize=(8, 6))
    properties = list(final_properties.keys())
    achieved = [final_properties[prop] for prop in properties]
    targets = [target_properties[prop] for prop in properties]

    x = range(len(properties))
    width = 0.35

    plt.bar(x, targets, width, label='Target')
    plt.bar([i + width for i in x], achieved, width, label='Achieved')

    plt.xlabel('Properties')
    plt.ylabel('Value')
    plt.title('Property Comparison')
    plt.xticks([i + width / 2 for i in x], properties)
    plt.legend()
    plt.savefig(output_dir / 'property_comparison.png')
    plt.close()

    return {
        'type': 'properties',
        'data': {
            'final_properties': final_properties,
            'target_properties': target_properties,
            'match_scores': calculate_match_scores(final_properties, target_properties)
        }
    }


async def analyze_structures(results: Dict, output_dir: Path) -> Dict:
    """구조 분석"""
    initial_structure = results['optimization_path']['initial_structure']
    final_structure = results['final_structure']

    # 구조 변화 분석
    structural_changes = analyze_structural_changes(
        initial_structure,
        final_structure
    )

    # 구조 변화 시각화
    plot_structural_changes(structural_changes, output_dir / 'structure_changes.png')

    return {
        'type': 'structure',
        'data': {
            'structural_changes': structural_changes,
            'stability_metrics': analyze_stability(final_structure)
        }
    }


async def analyze_performance(results: Dict, output_dir: Path) -> Dict:
    """성능 분석"""
    metrics = results['metrics']

    # 성능 메트릭스 저장
    performance_data = {
        'computation_time': metrics['computation_time'],
        'total_iterations': metrics['total_iterations'],
        'final_score': metrics['final_score'],
        'efficiency': calculate_efficiency(metrics)
    }

    # 성능 시각화
    plot_performance_metrics(performance_data, output_dir / 'performance.png')

    return {
        'type': 'performance',
        'data': performance_data
    }


async def generate_report(analysis_results: List[Dict], output_dir: Path):
    """종합 보고서 생성"""
    report = {
        'summary': {
            'timestamp': datetime.now().isoformat(),
            'analyses_performed': [result['type'] for result in analysis_results]
        },
        'results': {
            result['type']: result['data']
            for result in analysis_results
        }
    }

    # 보고서 저장
    with open(output_dir / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # HTML 보고서 생성
    generate_html_report(report, output_dir / 'analysis_report.html')


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Analyze optimization results")

    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to results directory"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to output directory"
    )

    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["all", "convergence", "property", "structure", "performance"],
        default="all",
        help="Type of analysis to perform"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # 비동기 실행
    asyncio.run(
        analyze_results(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            analysis_type=args.analysis_type,
            config_path=args.config,
            verbose=args.verbose
        )
    )


if __name__ == "__main__":
    main()