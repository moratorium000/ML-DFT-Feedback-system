from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from ase import Atoms, Atom
from ase.io import read, write
import spglib
from pymatgen.core import Structure as PMGStructure
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter

from core.interfaces import Structure


class StructureConverter:
    """구조 형식 변환기"""

    @staticmethod
    def to_ase(structure: Structure) -> Atoms:
        """Structure를 ASE Atoms로 변환"""
        atoms = Atoms(
            numbers=structure.atomic_numbers,
            positions=structure.positions,
            cell=structure.lattice_vectors,
            pbc=True
        )

        if structure.constraints is not None:
            atoms.set_constraint(structure.constraints)

        return atoms

    @staticmethod
    def from_ase(atoms: Atoms) -> Structure:
        """ASE Atoms를 Structure로 변환"""
        return Structure(
            atomic_numbers=atoms.numbers,
            positions=atoms.get_scaled_positions(),
            lattice_vectors=atoms.cell.array,
            cell_params={"a": atoms.cell.lengths()[0],
                         "b": atoms.cell.lengths()[1],
                         "c": atoms.cell.lengths()[2],
                         "alpha": atoms.cell.angles()[0],
                         "beta": atoms.cell.angles()[1],
                         "gamma": atoms.cell.angles()[2]},
            formula=atoms.get_chemical_formula(),
            space_group=spglib.get_spacegroup(atoms),
            constraints=atoms.constraints
        )

    @staticmethod
    def to_pymatgen(structure: Structure) -> PMGStructure:
        """Structure를 Pymatgen Structure로 변환"""
        lattice = structure.lattice_vectors
        species = []
        coords = []

        for z, pos in zip(structure.atomic_numbers, structure.positions):
            species.append(PeriodicTable.get_symbol(z))
            coords.append(pos)

        return PMGStructure(
            lattice=lattice,
            species=species,
            coords=coords,
            coords_are_cartesian=False
        )

    @staticmethod
    def from_pymatgen(structure: PMGStructure) -> Structure:
        """Pymatgen Structure를 Structure로 변환"""
        return Structure(
            atomic_numbers=[PeriodicTable.get_z(str(site.specie))
                            for site in structure.sites],
            positions=structure.frac_coords,
            lattice_vectors=structure.lattice.matrix,
            cell_params={"a": structure.lattice.a,
                         "b": structure.lattice.b,
                         "c": structure.lattice.c,
                         "alpha": structure.lattice.alpha,
                         "beta": structure.lattice.beta,
                         "gamma": structure.lattice.gamma},
            formula=structure.composition.reduced_formula,
            space_group=structure.get_space_group_info()[0]
        )


class InputConverter:
    """DFT 입력 형식 변환기"""

    def to_vasp(self, structure: Structure, params: Dict) -> Dict[str, str]:
        """VASP 입력 파일 생성"""
        # POSCAR 생성
        poscar = self._generate_poscar(structure)

        # INCAR 생성
        incar = self._generate_incar(params)

        # KPOINTS 생성
        kpoints = self._generate_kpoints(params.get('kpoints', [1, 1, 1]))

        # POTCAR 생성
        potcar = self._generate_potcar(structure.atomic_numbers)

        return {
            'POSCAR': poscar,
            'INCAR': incar,
            'KPOINTS': kpoints,
            'POTCAR': potcar
        }

    def to_qe(self, structure: Structure, params: Dict) -> str:
        """QE 입력 파일 생성"""
        # 기본 설정
        input_str = "&CONTROL\n"
        input_str += self._dict_to_qe(params.get('control', {}))

        # 시스템 설정
        input_str += "&SYSTEM\n"
        input_str += self._dict_to_qe(params.get('system', {}))

        # 전자 설정
        input_str += "&ELECTRONS\n"
        input_str += self._dict_to_qe(params.get('electrons', {}))

        # 원자 구조
        input_str += "ATOMIC_SPECIES\n"
        input_str += self._generate_atomic_species(structure)

        # 격자 벡터
        input_str += "CELL_PARAMETERS angstrom\n"
        input_str += self._generate_cell_parameters(structure)

        # 원자 위치
        input_str += "ATOMIC_POSITIONS crystal\n"
        input_str += self._generate_atomic_positions(structure)

        # k-points
        input_str += "K_POINTS automatic\n"
        input_str += self._generate_kpoints_qe(params.get('kpoints', [1, 1, 1]))

        return input_str

    def to_siesta(self, structure: Structure, params: Dict) -> str:
        """SIESTA 입력 파일 생성"""
        # SIESTA 입력 파일 형식으로 변환
        input_str = ""

        # 시스템 설정
        input_str += "SystemName\t{}\n".format(structure.formula)
        input_str += "SystemLabel\tsiesta\n\n"

        # 계산 설정
        input_str += self._dict_to_siesta(params)

        # 격자 정보
        input_str += "%block LatticeVectors\n"
        for vec in structure.lattice_vectors:
            input_str += "{:.6f} {:.6f} {:.6f}\n".format(*vec)
        input_str += "%endblock LatticeVectors\n\n"

        # 원자 위치
        input_str += "%block AtomicCoordinatesAndAtomicSpecies\n"
        for z, pos in zip(structure.atomic_numbers, structure.positions):
            input_str += "{:.6f} {:.6f} {:.6f} {}\n".format(*pos, z)
        input_str += "%endblock AtomicCoordinatesAndAtomicSpecies\n"

        return input_str

    def _generate_poscar(self, structure: Structure) -> str:
        """POSCAR 파일 생성"""
        pmg_structure = StructureConverter.to_pymatgen(structure)
        poscar = Poscar(pmg_structure)
        return str(poscar)

    def _generate_incar(self, params: Dict) -> str:
        """INCAR 파일 생성"""
        incar_str = ""
        for key, value in params.items():
            if isinstance(value, bool):
                value = ".TRUE." if value else ".FALSE."
            incar_str += f"{key} = {value}\n"
        return incar_str

    def _dict_to_qe(self, params: Dict) -> str:
        """딕셔너리를 QE 입력 형식으로 변환"""
        qe_str = ""
        for key, value in params.items():
            if isinstance(value, bool):
                value = ".true." if value else ".false."
            qe_str += f"  {key} = {value}\n"
        return qe_str + "/\n\n"

    def _dict_to_siesta(self, params: Dict) -> str:
        """딕셔너리를 SIESTA 입력 형식으로 변환"""
        siesta_str = ""
        for key, value in params.items():
            if isinstance(value, bool):
                value = "T" if value else "F"
            siesta_str += f"{key}\t{value}\n"
        return siesta_str + "\n"


class PeriodicTable:
    """원소 기호 <-> 원자 번호 변환"""
    _symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', ...]  # 전체 원소 기호

    @classmethod
    def get_symbol(cls, z: int) -> str:
        """원자 번호로 원소 기호 얻기"""
        return cls._symbols[z - 1]

    @classmethod
    def get_z(cls, symbol: str) -> int:
        """원소 기호로 원자 번호 얻기"""
        return cls._symbols.index(symbol) + 1