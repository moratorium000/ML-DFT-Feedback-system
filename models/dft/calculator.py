from typing import Dict, List, Optional, Union
from pathlib import Path
import asyncio
import numpy as np
from enum import Enum
import logging

from core.interfaces import Structure, DFTResult, CalculationStatus
from core.protocols import IDFTCalculator
from core.utils.logger import get_logger


class DFTCode(Enum):
    """지원하는 DFT 코드"""
    VASP = "vasp"
    QE = "quantum-espresso"
    SIESTA = "siesta"


class DFTCalculator(IDFTCalculator):
    """DFT 계산 관리자"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger(__name__)
        self.work_dir = Path(config['work_dir'])
        self.dft_code = DFTCode(config['dft_code'])
        self.calculator = self._initialize_calculator()
        self.active_jobs: Dict[str, CalculationStatus] = {}

    async def calculate(self,
                        structure: Structure,
                        calc_params: Optional[Dict] = None) -> DFTResult:
        """DFT 계산 실행"""
        try:
            # 작업 디렉토리 준비
            calc_dir = self._prepare_calculation_dir(structure)

            # 입력 파일 생성
            input_files = self._prepare_input_files(
                structure,
                calc_params or self.config['default_params']
            )

            # 계산 실행
            job_id = await self._submit_calculation(calc_dir, input_files)
            self.active_jobs[job_id] = CalculationStatus.RUNNING

            # 계산 모니터링
            result = await self._monitor_calculation(job_id)

            # 결과 처리
            if result.convergence:
                return self._process_results(result, structure)
            else:
                raise CalculationNotConvergedError(
                    f"Calculation {job_id} did not converge"
                )

        except Exception as e:
            self.logger.error(f"DFT calculation failed: {e}")
            raise

    async def check_convergence(self,
                                calc_result: DFTResult) -> Dict[str, bool]:
        """수렴성 검사"""
        energy_conv = self._check_energy_convergence(calc_result)
        force_conv = self._check_force_convergence(calc_result)
        electronic_conv = self._check_electronic_convergence(calc_result)

        return {
            'energy_converged': energy_conv,
            'force_converged': force_conv,
            'electronic_converged': electronic_conv,
            'all_converged': all([energy_conv, force_conv, electronic_conv])
        }

    async def estimate_cost(self,
                            structure: Structure,
                            calc_params: Dict) -> Dict[str, float]:
        """계산 비용 추정"""
        n_atoms = len(structure.atomic_numbers)
        n_electrons = self._count_valence_electrons(structure)
        k_points = np.prod(calc_params.get('kpoints', [1, 1, 1]))

        # 계산 비용 예측 모델
        cpu_hours = self._estimate_cpu_hours(
            n_atoms,
            n_electrons,
            k_points,
            calc_params
        )

        memory_gb = self._estimate_memory_requirement(
            n_atoms,
            n_electrons,
            k_points,
            calc_params
        )

        return {
            'cpu_hours': cpu_hours,
            'memory_gb': memory_gb,
            'disk_space_gb': memory_gb * 3  # 대략적인 추정
        }

    def _initialize_calculator(self):
        """DFT 코드별 계산기 초기화"""
        if self.dft_code == DFTCode.VASP:
            return VASPCalculator(self.config)
        elif self.dft_code == DFTCode.QE:
            return QECalculator(self.config)
        elif self.dft_code == DFTCode.SIESTA:
            return SIESTACalculator(self.config)
        else:
            raise ValueError(f"Unsupported DFT code: {self.dft_code}")

    def _prepare_calculation_dir(self, structure: Structure) -> Path:
        """계산 디렉토리 준비"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calc_dir = self.work_dir / f"calc_{timestamp}"
        calc_dir.mkdir(parents=True, exist_ok=True)
        return calc_dir

    def _prepare_input_files(self,
                             structure: Structure,
                             params: Dict) -> Dict[str, Path]:
        """입력 파일 생성"""
        return self.calculator.prepare_input_files(structure, params)

    async def _submit_calculation(self,
                                  calc_dir: Path,
                                  input_files: Dict[str, Path]) -> str:
        """계산 작업 제출"""
        return await self.calculator.submit_job(calc_dir, input_files)

    async def _monitor_calculation(self, job_id: str) -> DFTResult:
        """계산 모니터링"""
        while True:
            status = await self.calculator.check_job_status(job_id)

            if status == CalculationStatus.COMPLETED:
                return await self.calculator.collect_results(job_id)
            elif status == CalculationStatus.FAILED:
                raise CalculationFailedError(f"Calculation {job_id} failed")

            await asyncio.sleep(self.config['polling_interval'])

    def _process_results(self,
                         raw_result: Dict,
                         initial_structure: Structure) -> DFTResult:
        """결과 처리"""
        return DFTResult(
            initial_structure=initial_structure,
            final_structure=self._parse_final_structure(raw_result),
            total_energy=raw_result['total_energy'],
            energy_per_atom=raw_result['total_energy'] / len(initial_structure.atomic_numbers),
            forces=raw_result['forces'],
            stress_tensor=raw_result['stress'],
            band_gap=raw_result.get('band_gap'),
            dos=raw_result.get('dos'),
            convergence=True,
            calculation_time=raw_result['calculation_time'],
            error_messages=[],
            additional_properties=raw_result.get('additional_properties', {})
        )

    def _check_energy_convergence(self, calc_result: DFTResult) -> bool:
        """에너지 수렴성 검사"""
        threshold = self.config['convergence_criteria']['energy']
        energy_diff = calc_result.energy_per_atom - calc_result.formation_energy
        return abs(energy_diff) < threshold

    def _check_force_convergence(self, calc_result: DFTResult) -> bool:
        """힘 수렴성 검사"""
        threshold = self.config['convergence_criteria']['force']
        max_force = np.max(np.abs(calc_result.forces))
        return max_force < threshold

    def _check_electronic_convergence(self, calc_result: DFTResult) -> bool:
        """전자 구조 수렴성 검사"""
        return calc_result.additional_properties.get('electronic_converged', False)


class CalculationFailedError(Exception):
    pass


class CalculationNotConvergedError(Exception):
    pass