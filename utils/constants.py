from enum import Enum
from typing import Any, Dict, Final

# 시스템 상수
MAX_RETRIES: Final[int] = 3
DEFAULT_TIMEOUT: Final[int] = 30  # seconds
CHUNK_SIZE: Final[int] = 8192     # bytes
CACHE_TTL: Final[int] = 3600      # seconds

# 물리 상수
ATOMIC_UNITS: Final[Dict[str, float]] = {
    'energy': 27.211386245988,     # eV
    'length': 0.529177210903,      # Å
    'force': 51.42208619083,       # eV/Å
    'time': 2.418884326509e-17,    # s
    'mass': 9.1093837015e-31,      # kg
    'charge': 1.602176634e-19      # C
}

# 원자 관련 상수
MIN_ATOMIC_DISTANCE: Final[float] = 0.7    # Å
MAX_ATOMIC_DISTANCE: Final[float] = 3.0    # Å
MIN_CELL_ANGLE: Final[float] = 30.0       # degrees
MAX_CELL_ANGLE: Final[float] = 150.0      # degrees
MAX_VOLUME_CHANGE: Final[float] = 0.3     # 30%

# 원자 질량 (amu)
ATOMIC_MASS: Final[Dict[int, float]] = {
    1: 1.008,    # H
    2: 4.003,    # He
    3: 6.941,    # Li
    4: 9.012,    # Be
    5: 10.81,    # B
    6: 12.011,   # C
    7: 14.007,   # N
    8: 15.999,   # O
    9: 18.998,   # F
    10: 20.180,  # Ne
    11: 22.990,  # Na
    12: 24.305,  # Mg
    13: 26.982,  # Al
    14: 28.086,  # Si
    15: 30.974,  # P
    16: 32.065,  # S
    17: 35.453,  # Cl
    18: 39.948,  # Ar
    19: 39.098,  # K
    20: 40.078,  # Ca
    21: 44.956,  # Sc
    22: 47.867,  # Ti
    23: 50.942,  # V
    24: 51.996,  # Cr
    25: 54.938,  # Mn
    26: 55.845,  # Fe
    27: 58.933,  # Co
    28: 58.693,  # Ni
    29: 63.546,  # Cu
    30: 65.38,   # Zn
    31: 69.723,  # Ga
    32: 72.64,   # Ge
    33: 74.922,  # As
    34: 78.96,   # Se
    35: 79.904,  # Br
    36: 83.798,  # Kr
    37: 85.468,  # Rb
    38: 87.62,   # Sr
    39: 88.906,  # Y
    40: 91.224,  # Zr
    41: 92.906,  # Nb
    42: 95.96,   # Mo
    44: 101.07,  # Ru
    45: 102.91,  # Rh
    46: 106.42,  # Pd
    47: 107.87,  # Ag
    48: 112.41,  # Cd
    49: 114.82,  # In
    50: 118.71,  # Sn
    51: 121.76,  # Sb
    52: 127.60,  # Te
    53: 126.90,  # I
    54: 131.29,  # Xe
    55: 132.91,  # Cs
    56: 137.33,  # Ba
    57: 138.91,  # La
    72: 178.49,  # Hf
    73: 180.95,  # Ta
    74: 183.84,  # W
    75: 186.21,  # Re
    76: 190.23,  # Os
    77: 192.22,  # Ir
    78: 195.08,  # Pt
    79: 196.97,  # Au
    80: 200.59,  # Hg
    81: 204.38,  # Tl
    82: 207.2,   # Pb
    83: 208.98,  # Bi
}

class CalculationStatus(str, Enum):
    """계산 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CONVERGENCE_FAILED = "convergence_failed"

class DataFormat(str, Enum):
    """데이터 형식"""
    JSON = "json"
    YAML = "yaml"
    HDF5 = "hdf5"
    CIF = "cif"
    POSCAR = "poscar"
    XYZ = "xyz"

class ErrorCode(int, Enum):
    """오류 코드"""
    SUCCESS = 0
    INVALID_INPUT = 100
    CALCULATION_FAILED = 200
    CONVERGENCE_ERROR = 201
    TIMEOUT_ERROR = 202
    STORAGE_ERROR = 300
    DATABASE_ERROR = 400
    UNKNOWN_ERROR = 999

# DFT 계산 관련 상수
DFT_CONVERGENCE_CRITERIA: Final[Dict[str, float]] = {
    'energy': 1e-5,    # eV
    'force': 0.02,     # eV/Å
    'stress': 0.1,     # GPa
    'density': 1e-6    # e/Å³
}

DFT_DEFAULT_PARAMETERS: Final[Dict[str, Any]] = {
    'xc_functional': 'PBE',
    'energy_cutoff': 520,    # eV
    'kpoints': [2, 2, 2],
    'smearing': 'gaussian',
    'sigma': 0.05,          # eV
    'mixing_beta': 0.7,
    'max_iterations': 100
}

# ML 관련 상수
ML_DEFAULT_PARAMETERS: Final[Dict[str, Any]] = {
    'hidden_layers': [256, 128, 64],
    'activation': 'relu',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping': {
        'patience': 10,
        'min_delta': 0.001
    }
}

# 파일 시스템 관련 상수
FILE_PERMISSIONS: Final[Dict[str, int]] = {
    'default': 0o644,
    'executable': 0o755,
    'private': 0o600
}

ALLOWED_FILE_EXTENSIONS: Final[tuple] = (
    '.json', '.yaml', '.hdf5', '.cif',
    '.poscar', '.xyz', '.dat'
)

# 캐시 관련 상수
CACHE_SETTINGS: Final[Dict[str, Any]] = {
    'max_size': 1000,
    'ttl': 3600,
    'cleanup_interval': 300
}

# 데이터베이스 관련 상수
DB_SETTINGS: Final[Dict[str, Any]] = {
    'max_connections': 10,
    'connection_timeout': 30,
    'pool_recycle': 3600,
    'pool_size': 5
}

# 로깅 관련 상수
LOG_SETTINGS: Final[Dict[str, Any]] = {
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    'date_format': "%Y-%m-%d %H:%M:%S"
}

# HTTP 관련 상수
HTTP_SETTINGS: Final[Dict[str, Any]] = {
    'timeout': 30,
    'max_retries': 3,
    'backoff_factor': 2
}

# 보안 관련 상수
SECURITY_SETTINGS: Final[Dict[str, Any]] = {
    'key_length': 32,
    'salt_length': 16,
    'iterations': 100000,
    'hash_algorithm': 'sha256'
}