from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import json
from utils.constants import *


@dataclass
class DFTSettings:
    """DFT 계산 설정"""
    code: str = "vasp"  # DFT 코드 선택
    parameters: Dict = field(default_factory=lambda: DFT_DEFAULT_PARAMETERS)
    convergence_criteria: Dict = field(default_factory=lambda: DFT_CONVERGENCE_CRITERIA)
    max_iterations: int = 100
    parallel_settings: Dict = field(default_factory=lambda: {
        'ncore': 4,
        'kpar': 2
    })


@dataclass
class MLSettings:
    """ML 모델 설정"""
    model_parameters: Dict = field(default_factory=lambda: ML_DEFAULT_PARAMETERS)
    training_parameters: Dict = field(default_factory=lambda: {
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2
    })
    device: str = 'cuda'  # or 'cpu'
    checkpoint_dir: Path = Path("checkpoints")


@dataclass
class DatabaseSettings:
    """데이터베이스 설정"""
    url: str = "sqlite:///mldft.db"
    settings: Dict = field(default_factory=lambda: DB_SETTINGS)


@dataclass
class StorageSettings:
    """저장소 설정"""
    base_dir: Path = Path("storage")
    backup_dir: Optional[Path] = Path("backup")
    max_size_gb: float = 100.0
    compression: bool = True
    file_permissions: int = FILE_PERMISSIONS['default']


@dataclass
class CacheSettings:
    """캐시 설정"""
    enabled: bool = True
    settings: Dict = field(default_factory=lambda: CACHE_SETTINGS)


@dataclass
class LogSettings:
    """로깅 설정"""
    level: str = "INFO"
    settings: Dict = field(default_factory=lambda: LOG_SETTINGS)


@dataclass
class PrototypeSettings:
    """Prototype 설정"""
    cache_dir: Path = Path(".cache/prototypes")
    validation_settings: Dict = field(default_factory=lambda: {
        'min_atomic_distance': 0.7,
        'max_atomic_distance': 3.0,
        'min_cell_angle': 30.0,
        'max_cell_angle': 150.0,
        'max_volume_change': 0.3
    })
    data_format: str = "json"
    backup_enabled: bool = True
    max_cache_size: int = 1000


@dataclass
class PathSettings:
    """경로 최적화 설정"""
    optimization_parameters: Dict = field(default_factory=lambda: {
        'max_iterations': 100,
        'convergence_threshold': 0.01,
        'step_size': 0.1,
        'momentum': 0.9
    })
    mutation_settings: Dict = field(default_factory=lambda: {
        'mutation_rate': 0.3,
        'crossover_rate': 0.7,
        'population_size': 50
    })
    dft_validation_interval: int = 5


@dataclass
class SystemSettings:
    """시스템 전체 설정"""
    dft: DFTSettings = field(default_factory=DFTSettings)
    ml: MLSettings = field(default_factory=MLSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    logging: LogSettings = field(default_factory=LogSettings)
    prototype: PrototypeSettings = field(default_factory=PrototypeSettings)
    path: PathSettings = field(default_factory=PathSettings)

    # 최적화 관련 설정
    max_iterations: int = 100
    convergence_window: int = 5
    convergence_threshold: float = 0.95
    min_change_threshold: float = 0.001

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'SystemSettings':
        """파일에서 설정 로드"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.suffix == '.yaml':
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path) as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return cls(**config_dict)

    def save(self, config_path: Union[str, Path]):
        """설정을 파일로 저장"""
        config_path = Path(config_path)
        config_dict = {
            'dft': self.dft.__dict__,
            'ml': self.ml.__dict__,
            'database': self.database.__dict__,
            'storage': self.storage.__dict__,
            'cache': self.cache.__dict__,
            'logging': self.logging.__dict__,
            'prototype': self.prototype.__dict__,
            'path': self.path.__dict__,
            'max_iterations': self.max_iterations,
            'convergence_window': self.convergence_window,
            'convergence_threshold': self.convergence_threshold,
            'min_change_threshold': self.min_change_threshold
        }

        if config_path.suffix == '.yaml':
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def validate(self) -> List[str]:
        """설정 유효성 검사"""
        errors = []

        # DFT 설정 검사
        if self.dft.code not in ['vasp', 'qe', 'siesta']:
            errors.append(f"Unsupported DFT code: {self.dft.code}")

        # ML 설정 검사
        if not self.ml.checkpoint_dir.parent.exists():
            errors.append(
                f"Checkpoint directory parent does not exist: {self.ml.checkpoint_dir}"
            )

        # 저장소 설정 검사
        if self.storage.max_size_gb <= 0:
            errors.append(
                f"Invalid storage size: {self.storage.max_size_gb} GB"
            )

        return errors


def load_config(config_path: Optional[Union[str, Path]] = None) -> SystemSettings:
    """설정 로드"""
    if config_path is None:
        # 기본 설정 반환
        return SystemSettings()

    try:
        settings = SystemSettings.from_file(config_path)

        # 유효성 검사
        errors = settings.validate()
        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Invalid configuration:\n{error_msg}")

        return settings

    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")