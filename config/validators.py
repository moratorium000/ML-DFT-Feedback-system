from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import numpy as np
from pydantic import BaseModel, validator, Field
from enum import Enum
import re


class ConfigValidationError(Exception):
    """설정 검증 오류"""
    pass


class DFTCode(str, Enum):
    """지원하는 DFT 코드"""
    VASP = "vasp"
    QE = "quantum-espresso"
    SIESTA = "siesta"


class DFTValidator(BaseModel):
    """DFT 설정 검증"""
    code: DFTCode
    parameters: Dict[str, Any]
    parallel_settings: Dict[str, int]

    @validator('parameters')
    def validate_parameters(cls, v):
        """DFT 파라미터 검증"""
        required_fields = {
            'xc_functional': str,
            'energy_cutoff': (int, float),
            'kpoints': list
        }

        for field, field_type in required_fields.items():
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(v[field], field_type):
                raise ValueError(f"Invalid type for {field}")

        # 에너지 컷오프 범위 검사
        if v['energy_cutoff'] < 100 or v['energy_cutoff'] > 1000:
            raise ValueError("Energy cutoff should be between 100 and 1000 eV")

        # k-points 검사
        if not all(isinstance(k, int) and k > 0 for k in v['kpoints']):
            raise ValueError("k-points must be positive integers")

        return v

    @validator('parallel_settings')
    def validate_parallel(cls, v):
        """병렬 설정 검증"""
        if 'ncore' not in v or v['ncore'] < 1:
            raise ValueError("Invalid ncore value")
        return v


class MLValidator(BaseModel):
    """ML 설정 검증"""
    model_type: str
    hidden_layers: List[int]
    activation: str
    learning_rate: float = Field(gt=0, lt=1)
    batch_size: int = Field(gt=0)

    @validator('hidden_layers')
    def validate_layers(cls, v):
        """레이어 구성 검증"""
        if not v:
            raise ValueError("At least one hidden layer required")
        if not all(layer > 0 for layer in v):
            raise ValueError("Layer sizes must be positive")
        return v

    @validator('activation')
    def validate_activation(cls, v):
        """활성화 함수 검증"""
        valid_activations = {'relu', 'tanh', 'sigmoid', 'elu'}
        if v not in valid_activations:
            raise ValueError(f"Unsupported activation function: {v}")
        return v


class DatabaseValidator(BaseModel):
    """데이터베이스 설정 검증"""
    url: str
    max_connections: int = Field(gt=0)
    timeout: int = Field(gt=0)

    @validator('url')
    def validate_url(cls, v):
        """데이터베이스 URL 검증"""
        url_pattern = r'^(sqlite|postgresql|mysql)://.*'
        if not re.match(url_pattern, v):
            raise ValueError("Invalid database URL format")
        return v


class StorageValidator(BaseModel):
    """저장소 설정 검증"""
    base_dir: Path
    max_size_gb: float = Field(gt=0)
    compression: bool
    file_permissions: int

    @validator('base_dir')
    def validate_base_dir(cls, v):
        """디렉토리 경로 검증"""
        if not v.parent.exists():
            raise ValueError(f"Parent directory does not exist: {v.parent}")
        return v

    @validator('file_permissions')
    def validate_permissions(cls, v):
        """파일 권한 검증"""
        if not 0o000 <= v <= 0o777:
            raise ValueError("Invalid file permissions")
        return v


class SystemValidator:
    """시스템 설정 검증"""

    def __init__(self):
        self.validators = {
            'dft': DFTValidator,
            'ml': MLValidator,
            'database': DatabaseValidator,
            'storage': StorageValidator
        }

    def validate(self, config: Dict) -> List[str]:
        """설정 검증"""
        errors = []

        for section, validator in self.validators.items():
            if section not in config:
                errors.append(f"Missing section: {section}")
                continue

            try:
                validator(**config[section])
            except Exception as e:
                errors.append(f"Validation error in {section}: {str(e)}")

        return errors


def validate_config(config: Dict) -> List[str]:
    """설정 파일 검증"""
    validator = SystemValidator()
    return validator.validate(config)


def validate_paths(paths: Dict[str, Path]) -> List[str]:
    """경로 검증"""
    errors = []

    for name, path in paths.items():
        if not isinstance(path, Path):
            errors.append(f"Invalid path type for {name}: {type(path)}")
            continue

        if not path.parent.exists():
            errors.append(f"Parent directory does not exist for {name}: {path.parent}")

    return errors


def validate_dft_parameters(params: Dict) -> List[str]:
    """DFT 파라미터 검증"""
    validator = DFTValidator
    try:
        validator(code=params['code'],
                  parameters=params['parameters'],
                  parallel_settings=params['parallel_settings'])
        return []
    except Exception as e:
        return [str(e)]


def validate_ml_parameters(params: Dict) -> List[str]:
    """ML 파라미터 검증"""
    validator = MLValidator
    try:
        validator(**params)
        return []
    except Exception as e:
        return [str(e)]