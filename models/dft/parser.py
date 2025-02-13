from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import re

from core.interfaces import Structure, DFTResult
from core.utils.logger import get_logger


class DFTOutputParser:
    """DFT 출력 파일 파서"""

    def __init__(self, dft_code: str):
        self.dft_code = dft_code
        self.logger = get_logger(__name__)

    def parse_output(self, calc_dir: Path) -> Dict:
        """출력 파일 파싱"""
        if self.dft_code == "vasp":
            return VASPParser().parse(calc_dir)
        elif self.dft_code == "qe":
            return QEParser().parse(calc_dir)
        elif self.dft_code == "siesta":
            return SIESTAParser().parse(calc_dir)
        else:
            raise ValueError(f"Unsupported DFT code: {self.dft_code}")


class VASPParser:
    """VASP 출력 파서"""

    def parse(self, calc_dir: Path) -> Dict:
        """VASP 출력 파싱"""
        results = {}

        # OUTCAR 파싱
        results.update(self._parse_outcar(calc_dir / "OUTCAR"))

        # vasprun.xml 파싱
        results.update(self._parse_vasprun(calc_dir / "vasprun.xml"))

        # DOSCAR 파싱 (있는 경우)
        doscar_path = calc_dir / "DOSCAR"
        if doscar_path.exists():
            results['dos'] = self._parse_doscar(doscar_path)

        return results

    def _parse_outcar(self, outcar_path: Path) -> Dict:
        """OUTCAR 파일 파싱"""
        results = {
            'energies': [],
            'forces': [],
            'stress': [],
            'magnetic_moments': [],
            'convergence_electronic': False,
            'convergence_ionic': False
        }

        with open(outcar_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            # 에너지 추출
            if "free  energy   TOTEN" in line:
                energy = float(line.split()[-2])
                results['energies'].append(energy)

            # 힘 추출
            elif "TOTAL-FORCE" in line:
                forces = []
                for j in range(i + 2, i + 2 + self.n_atoms):
                    force_line = lines[j].split()
                    forces.append([float(x) for x in force_line[3:6]])
                results['forces'].append(np.array(forces))

            # 응력 텐서 추출
            elif "in kB" in line:
                stress = [float(x) for x in line.split()[2:8]]
                results['stress'].append(stress)

            # 수렴성 확인
            elif "reached required accuracy" in line:
                results['convergence_electronic'] = True

        return results

    def _parse_vasprun(self, vasprun_path: Path) -> Dict:
        """vasprun.xml 파일 파싱"""
        tree = ET.parse(vasprun_path)
        root = tree.getroot()

        results = {
            'parameters': {},
            'kpoints': [],
            'bands': [],
            'occupation': []
        }

        # 계산 파라미터 추출
        for param in root.findall(".//parameters/"):
            results['parameters'][param.attrib['name']] = param.text

        # k-points 추출
        for kpoint in root.findall(".//kpoints/varray[@name='kpointlist']/v"):
            results['kpoints'].append([float(x) for x in kpoint.text.split()])

        # 밴드 구조 추출
        for set in root.findall(".//eigenvalues/array/set/set"):
            bands = []
            for r in set.findall('r'):
                bands.append([float(x) for x in r.text.split()])
            results['bands'].append(bands)

        return results

    def _parse_doscar(self, doscar_path: Path) -> Dict:
        """DOSCAR 파일 파싱"""
        with open(doscar_path, 'r') as f:
            lines = f.readlines()

        # 헤더 정보 추출
        n_atoms = int(lines[0].split()[0])
        n_points = int(lines[5].split()[2])
        e_fermi = float(lines[5].split()[3])

        # DOS 데이터 추출
        energies = []
        total_dos = []
        for i in range(6, 6 + n_points):
            data = [float(x) for x in lines[i].split()]
            energies.append(data[0])
            total_dos.append(data[1])

        return {
            'n_points': n_points,
            'e_fermi': e_fermi,
            'energies': np.array(energies),
            'total_dos': np.array(total_dos)
        }


class QEParser:
    """Quantum ESPRESSO 출력 파서"""

    def parse(self, calc_dir: Path) -> Dict:
        """QE 출력 파싱"""
        results = {}

        # 출력 파일 파싱
        output_path = calc_dir / "*.out"  # 실제 파일명에 맞게 수정 필요
        results.update(self._parse_output(output_path))

        # XML 파일 파싱
        xml_path = calc_dir / "*.xml"  # 실제 파일명에 맞게 수정 필요
        if xml_path.exists():
            results.update(self._parse_xml(xml_path))

        return results

    def _parse_output(self, output_path: Path) -> Dict:
        """QE 출력 파일 파싱"""
        results = {
            'energies': [],
            'forces': [],
            'stress': [],
            'convergence': False
        }

        with open(output_path, 'r') as f:
            content = f.read()

        # 에너지 추출
        energy_pattern = r"!\s+total energy\s+=\s+([\d\.-]+)\s+Ry"
        energies = [float(e) for e in re.findall(energy_pattern, content)]
        results['energies'] = energies

        # 힘 추출
        force_pattern = r"Forces acting on atoms.*\n([\s\S]*?)\n\n"
        force_blocks = re.findall(force_pattern, content)
        for block in force_blocks:
            forces = []
            for line in block.split('\n'):
                if 'atom' in line:
                    force = [float(x) for x in line.split()[-3:]]
                    forces.append(force)
            results['forces'].append(np.array(forces))

        return results

    def _parse_xml(self, xml_path: Path) -> Dict:
        """QE XML 파일 파싱"""
        # XML 파싱 구현
        pass


class SIESTAParser:
    """SIESTA 출력 파서"""

    def parse(self, calc_dir: Path) -> Dict:
        """SIESTA 출력 파싱"""
        results = {}

        # 출력 파일 파싱
        output_path = calc_dir / "*.out"  # 실제 파일명에 맞게 수정 필요
        results.update(self._parse_output(output_path))

        return results

    def _parse_output(self, output_path: Path) -> Dict:
        """SIESTA 출력 파일 파싱"""
        # SIESTA 출력 파싱 구현
        pass


class OutputParserError(Exception):
    """파싱 오류"""
    pass