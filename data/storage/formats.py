from typing import Dict, List, Optional, Union, BinaryIO
import numpy as np
from pathlib import Path
from ase.io import read, write
from pymatgen.core import Structure as PMGStructure
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter
import json
import yaml


class FormatConverter:
    """구조 데이터 형식 변환기"""

    @staticmethod
    def to_dict(structure: Union['Structure', PMGStructure, 'Atoms']) -> Dict:
        """구조를 딕셔너리로 변환"""
        if isinstance(structure, PMGStructure):
            return FormatConverter._pmg_to_dict(structure)
        elif hasattr(structure, 'get_atomic_numbers'):  # ASE Atoms
            return FormatConverter._ase_to_dict(structure)
        else:
            return FormatConverter._structure_to_dict(structure)

    @staticmethod
    def from_dict(data: Dict) -> 'Structure':
        """딕셔너리에서 구조 생성"""
        from ...interfaces import Structure

        return Structure(
            atomic_numbers=np.array(data['atomic_numbers']),
            positions=np.array(data['positions']),
            lattice_vectors=np.array(data['lattice_vectors']),
            cell_params=data['cell_params'],
            formula=data['formula'],
            space_group=data.get('space_group'),
            charge=data.get('charge', 0.0),
            spin=data.get('spin', 0.0)
        )

    @staticmethod
    def to_ase(structure: 'Structure') -> 'Atoms':
        """ASE Atoms 형식으로 변환"""
        from ase import Atoms

        return Atoms(
            numbers=structure.atomic_numbers,
            positions=structure.positions,
            cell=structure.lattice_vectors,
            pbc=True
        )

    @staticmethod
    def to_pymatgen(structure: 'Structure') -> PMGStructure:
        """Pymatgen Structure 형식으로 변환"""
        return PMGStructure(
            lattice=structure.lattice_vectors,
            species=[FormatConverter._get_element(z)
                     for z in structure.atomic_numbers],
            coords=structure.positions,
            coords_are_cartesian=False
        )

    @staticmethod
    def write_file(structure: 'Structure',
                   filename: Union[str, Path],
                   format: str = None):
        """파일로 저장"""
        if format is None:
            format = Path(filename).suffix[1:]

        if format == 'json':
            FormatConverter._write_json(structure, filename)
        elif format == 'yaml':
            FormatConverter._write_yaml(structure, filename)
        elif format == 'cif':
            FormatConverter._write_cif(structure, filename)
        elif format in ['poscar', 'vasp']:
            FormatConverter._write_poscar(structure, filename)
        else:
            # ASE 지원 형식 사용
            ase_atoms = FormatConverter.to_ase(structure)
            write(filename, ase_atoms, format=format)

    @staticmethod
    def read_file(filename: Union[str, Path],
                  format: str = None) -> 'Structure':
        """파일에서 읽기"""
        if format is None:
            format = Path(filename).suffix[1:]

        if format == 'json':
            return FormatConverter._read_json(filename)
        elif format == 'yaml':
            return FormatConverter._read_yaml(filename)
        elif format in ['cif', 'poscar', 'vasp']:
            atoms = read(filename, format=format)
            return FormatConverter._ase_to_structure(atoms)
        else:
            atoms = read(filename, format=format)
            return FormatConverter._ase_to_structure(atoms)

    @staticmethod
    def _structure_to_dict(structure: 'Structure') -> Dict:
        """내부 Structure를 딕셔너리로 변환"""
        return {
            'atomic_numbers': structure.atomic_numbers.tolist(),
            'positions': structure.positions.tolist(),
            'lattice_vectors': structure.lattice_vectors.tolist(),
            'cell_params': structure.cell_params,
            'formula': structure.formula,
            'space_group': structure.space_group,
            'charge': getattr(structure, 'charge', 0.0),
            'spin': getattr(structure, 'spin', 0.0)
        }

    @staticmethod
    def _pmg_to_dict(structure: PMGStructure) -> Dict:
        """Pymatgen Structure를 딕셔너리로 변환"""
        return {
            'atomic_numbers': [s.Z for s in structure.species],
            'positions': structure.frac_coords.tolist(),
            'lattice_vectors': structure.lattice.matrix.tolist(),
            'cell_params': {
                'a': structure.lattice.a,
                'b': structure.lattice.b,
                'c': structure.lattice.c,
                'alpha': structure.lattice.alpha,
                'beta': structure.lattice.beta,
                'gamma': structure.lattice.gamma
            },
            'formula': structure.composition.reduced_formula,
            'space_group': structure.get_space_group_info()[0]
        }

    @staticmethod
    def _ase_to_dict(atoms: 'Atoms') -> Dict:
        """ASE Atoms를 딕셔너리로 변환"""
        return {
            'atomic_numbers': atoms.numbers.tolist(),
            'positions': atoms.get_scaled_positions().tolist(),
            'lattice_vectors': atoms.cell.array.tolist(),
            'cell_params': {
                'a': atoms.cell.lengths()[0],
                'b': atoms.cell.lengths()[1],
                'c': atoms.cell.lengths()[2],
                'alpha': atoms.cell.angles()[0],
                'beta': atoms.cell.angles()[1],
                'gamma': atoms.cell.angles()[2]
            },
            'formula': atoms.get_chemical_formula()
        }

    @staticmethod
    def _ase_to_structure(atoms: 'Atoms') -> 'Structure':
        """ASE Atoms에서 Structure 생성"""
        data = FormatConverter._ase_to_dict(atoms)
        return FormatConverter.from_dict(data)

    @staticmethod
    def _write_json(structure: 'Structure', filename: Union[str, Path]):
        """JSON 형식으로 저장"""
        data = FormatConverter.to_dict(structure)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _write_yaml(structure: 'Structure', filename: Union[str, Path]):
        """YAML 형식으로 저장"""
        data = FormatConverter.to_dict(structure)
        with open(filename, 'w') as f:
            yaml.dump(data, f)

    @staticmethod
    def _write_cif(structure: 'Structure', filename: Union[str, Path]):
        """CIF 형식으로 저장"""
        pmg_structure = FormatConverter.to_pymatgen(structure)
        CifWriter(pmg_structure).write_file(filename)

    @staticmethod
    def _write_poscar(structure: 'Structure', filename: Union[str, Path]):
        """POSCAR 형식으로 저장"""
        pmg_structure = FormatConverter.to_pymatgen(structure)
        Poscar(pmg_structure).write_file(filename)