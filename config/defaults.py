from typing import Dict, Final
from pathlib import Path

# 시스템 기본 경로
DEFAULT_PATHS: Final[Dict[str, Path]] = {
    'workspace': Path("./mldft_workspace"),
    'cache': Path("./cache"),
    'logs': Path("./logs"),
    'data': Path("./data"),
    'models': Path("./models"),
    'temp': Path("./temp")
}

# DFT 기본 설정
DEFAULT_DFT_SETTINGS: Final[Dict[str, any]] = {
    'code': 'vasp',
    'parameters': {
        'xc_functional': 'PBE',
        'energy_cutoff': 520,    # eV
        'kpoints': [2, 2, 2],
        'smearing': 'gaussian',
        'sigma': 0.05,          # eV
        'mixing_beta': 0.7,
        'max_iterations': 100,
        'electronic_steps': 60,
        'ionic_steps': 40,
        'force_convergence': 0.02,  # eV/Å
        'energy_convergence': 1e-5,  # eV
        'stress_convergence': 0.1,   # GPa
    },
    'parallel': {
        'ncore': 4,
        'kpar': 2,
        'lplane': True,
        'npar': 4
    }
}

# ML 기본 설정
DEFAULT_ML_SETTINGS: Final[Dict[str, any]] = {
    'model': {
        'architecture': 'graph',
        'hidden_layers': [256, 128, 64],
        'activation': 'relu',
        'dropout_rate': 0.1,
        'batch_norm': True
    },
    'training': {
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2,
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001
        },
        'scheduler': {
            'type': 'reduce_on_plateau',
            'factor': 0.5,
            'patience': 5
        }
    },
    'prediction': {
        'ensemble_size': 5,
        'uncertainty_method': 'dropout',
        'confidence_threshold': 0.8
    }
}

# 데이터베이스 기본 설정
DEFAULT_DB_SETTINGS: Final[Dict[str, any]] = {
    'url': 'sqlite:///mldft.db',
    'pool_size': 5,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'echo': False
}

# 저장소 기본 설정
DEFAULT_STORAGE_SETTINGS: Final[Dict[str, any]] = {
    'backend': 'local',
    'compression': True,
    'chunk_size': 8192,
    'max_size_gb': 100,
    'cleanup_threshold': 0.9,
    'file_permissions': 0o644,
    'backup': {
        'enabled': True,
        'interval': 24 * 3600,  # 24시간
        'keep_last': 5
    }
}

# 캐시 기본 설정
DEFAULT_CACHE_SETTINGS: Final[Dict[str, any]] = {
    'backend': 'memory',
    'max_size': 1000,
    'ttl': 3600,
    'cleanup_interval': 300,
    'structure_cache': {
        'max_size': 500,
        'ttl': 7200
    },
    'calculation_cache': {
        'max_size': 200,
        'ttl': 3600
    },
    'prediction_cache': {
        'max_size': 300,
        'ttl': 1800
    }
}

# 로깅 기본 설정
DEFAULT_LOG_SETTINGS: Final[Dict[str, any]] = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file': {
        'enabled': True,
        'max_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
        'encoding': 'utf-8'
    },
    'console': {
        'enabled': True,
        'color': True
    }
}

# 프로토타입 기본 설정
DEFAULT_PROTOTYPE_SETTINGS: Final[Dict[str, any]] = {
    'validation': {
        'min_atomic_distance': 0.7,  # Å
        'max_atomic_distance': 3.0,  # Å
        'min_cell_angle': 30.0,     # degrees
        'max_cell_angle': 150.0,    # degrees
        'max_volume_change': 0.3    # 30%
    },
    'cache': {
        'enabled': True,
        'max_size': 1000,
        'ttl': 7200
    }
}

# Mutation 기본 설정
DEFAULT_MUTATION_SETTINGS: Final[Dict[str, any]] = {
    'operators': {
        'displacement': {
            'probability': 0.3,
            'max_distance': 0.5  # Å
        },
        'substitution': {
            'probability': 0.1
        },
        'rotation': {
            'probability': 0.2,
            'max_angle': 30.0  # degrees
        },
        'strain': {
            'probability': 0.2,
            'max_strain': 0.1  # 10%
        },
        'coordination': {
            'probability': 0.2
        }
    },
    'constraints': {
        'preserve_stoichiometry': True,
        'preserve_symmetry': False,
        'maintain_coordination': True
    }
}

# 성능 기본 설정
DEFAULT_PERFORMANCE_SETTINGS: Final[Dict[str, any]] = {
    'max_workers': 4,
    'chunk_size': 1000,
    'timeout': 30,
    'retry_attempts': 3,
    'backoff_factor': 2
}