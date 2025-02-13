from typing import Dict, List, Optional, Union, BinaryIO
from pathlib import Path
import json
import numpy as np
import h5py
import yaml
from datetime import datetime
import aiofiles
import logging
from dataclasses import dataclass


@dataclass
class FileConfig:
    """파일 입출력 설정"""
    base_dir: Path
    backup_dir: Optional[Path] = None
    compression: bool = True
    chunk_size: int = 8192  # bytes
    file_permissions: int = 0o644


class FileIO:
    """파일 입출력 관리"""

    def __init__(self, config: FileConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 디렉토리 초기화
        self.config.base_dir.mkdir(parents=True, exist_ok=True)
        if self.config.backup_dir:
            self.config.backup_dir.mkdir(parents=True, exist_ok=True)

    async def save_structure(self,
                             structure_id: str,
                             data: Dict,
                             format: str = 'json') -> bool:
        """구조 데이터 저장"""
        try:
            file_path = self._get_structure_path(structure_id, format)

            if format == 'json':
                await self._save_json(file_path, data)
            elif format == 'hdf5':
                await self._save_hdf5(file_path, data)
            elif format == 'yaml':
                await self._save_yaml(file_path, data)
            else:
                raise ValueError(f"Unsupported format: {format}")

            return True

        except Exception as e:
            self.logger.error(f"Error saving structure {structure_id}: {e}")
            return False

    async def load_structure(self,
                             structure_id: str,
                             format: str = 'json') -> Optional[Dict]:
        """구조 데이터 로드"""
        try:
            file_path = self._get_structure_path(structure_id, format)

            if not file_path.exists():
                return None

            if format == 'json':
                return await self._load_json(file_path)
            elif format == 'hdf5':
                return await self._load_hdf5(file_path)
            elif format == 'yaml':
                return await self._load_yaml(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            self.logger.error(f"Error loading structure {structure_id}: {e}")
            return None

    async def save_calculation(self,
                               calc_id: str,
                               data: Dict) -> bool:
        """계산 결과 저장"""
        try:
            file_path = self._get_calculation_path(calc_id)

            # HDF5 형식으로 저장 (대용량 데이터 처리)
            with h5py.File(file_path, 'w') as f:
                # 기본 정보
                f.attrs['calc_id'] = calc_id
                f.attrs['created_at'] = datetime.now().isoformat()

                # 에너지 데이터
                energy_group = f.create_group('energies')
                for key, value in data.get('energies', {}).items():
                    energy_group.create_dataset(key, data=value)

                # 힘과 응력
                forces = data.get('forces')
                if forces is not None:
                    f.create_dataset('forces', data=forces)

                stress = data.get('stress')
                if stress is not None:
                    f.create_dataset('stress', data=stress)

                # 전자 구조 데이터
                if 'electronic' in data:
                    elec_group = f.create_group('electronic')
                    for key, value in data['electronic'].items():
                        if isinstance(value, np.ndarray):
                            elec_group.create_dataset(key, data=value)
                        else:
                            elec_group.attrs[key] = value

            return True

        except Exception as e:
            self.logger.error(f"Error saving calculation {calc_id}: {e}")
            return False

    async def load_calculation(self, calc_id: str) -> Optional[Dict]:
        """계산 결과 로드"""
        try:
            file_path = self._get_calculation_path(calc_id)

            if not file_path.exists():
                return None

            result = {}
            with h5py.File(file_path, 'r') as f:
                # 기본 정보
                result['calc_id'] = f.attrs['calc_id']
                result['created_at'] = f.attrs['created_at']

                # 에너지 데이터
                if 'energies' in f:
                    result['energies'] = {
                        key: value[...] for key, value in f['energies'].items()
                    }

                # 힘과 응력
                if 'forces' in f:
                    result['forces'] = f['forces'][...]
                if 'stress' in f:
                    result['stress'] = f['stress'][...]

                # 전자 구조 데이터
                if 'electronic' in f:
                    result['electronic'] = {
                        key: value[...] if isinstance(value, h5py.Dataset)
                        else value.attrs[key]
                        for key, value in f['electronic'].items()
                    }

            return result

        except Exception as e:
            self.logger.error(f"Error loading calculation {calc_id}: {e}")
            return None

    async def _save_json(self, path: Path, data: Dict):
        """JSON 형식 저장"""
        async with aiofiles.open(path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    async def _load_json(self, path: Path) -> Dict:
        """JSON 형식 로드"""
        async with aiofiles.open(path, 'r') as f:
            content = await f.read()
            return json.loads(content)

    async def _save_hdf5(self, path: Path, data: Dict):
        """HDF5 형식 저장"""
        with h5py.File(path, 'w') as f:
            self._write_dict_to_hdf5(f, data)

    async def _load_hdf5(self, path: Path) -> Dict:
        """HDF5 형식 로드"""
        with h5py.File(path, 'r') as f:
            return self._read_dict_from_hdf5(f)

    async def _save_yaml(self, path: Path, data: Dict):
        """YAML 형식 저장"""
        async with aiofiles.open(path, 'w') as f:
            await f.write(yaml.dump(data))

    async def _load_yaml(self, path: Path) -> Dict:
        """YAML 형식 로드"""
        async with aiofiles.open(path, 'r') as f:
            content = await f.read()
            return yaml.safe_load(content)

    def _get_structure_path(self, structure_id: str, format: str) -> Path:
        """구조 파일 경로"""
        return self.config.base_dir / 'structures' / f"{structure_id}.{format}"

    def _get_calculation_path(self, calc_id: str) -> Path:
        """계산 결과 파일 경로"""
        return self.config.base_dir / 'calculations' / f"{calc_id}.h5"