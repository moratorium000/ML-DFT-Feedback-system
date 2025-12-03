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

# 전기음성도 (Pauling scale)
ELECTRONEGATIVITY: Final[Dict[int, float]] = {
    1: 2.20,   # H
    2: 0.00,   # He
    3: 0.98,   # Li
    4: 1.57,   # Be
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    10: 0.00,  # Ne
    11: 0.93,  # Na
    12: 1.31,  # Mg
    13: 1.61,  # Al
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    18: 0.00,  # Ar
    19: 0.82,  # K
    20: 1.00,  # Ca
    21: 1.36,  # Sc
    22: 1.54,  # Ti
    23: 1.63,  # V
    24: 1.66,  # Cr
    25: 1.55,  # Mn
    26: 1.83,  # Fe
    27: 1.88,  # Co
    28: 1.91,  # Ni
    29: 1.90,  # Cu
    30: 1.65,  # Zn
    31: 1.81,  # Ga
    32: 2.01,  # Ge
    33: 2.18,  # As
    34: 2.55,  # Se
    35: 2.96,  # Br
    36: 3.00,  # Kr
    37: 0.82,  # Rb
    38: 0.95,  # Sr
    39: 1.22,  # Y
    40: 1.33,  # Zr
    41: 1.60,  # Nb
    42: 2.16,  # Mo
    44: 2.20,  # Ru
    45: 2.28,  # Rh
    46: 2.20,  # Pd
    47: 1.93,  # Ag
    48: 1.69,  # Cd
    49: 1.78,  # In
    50: 1.96,  # Sn
    51: 2.05,  # Sb
    52: 2.10,  # Te
    53: 2.66,  # I
    54: 2.60,  # Xe
}

# 원자 반경 (Angstrom, covalent radii)
ATOMIC_RADIUS: Final[Dict[int, float]] = {
    1: 0.31,   # H
    2: 0.28,   # He
    3: 1.28,   # Li
    4: 0.96,   # Be
    5: 0.84,   # B
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    9: 0.57,   # F
    10: 0.58,  # Ne
    11: 1.66,  # Na
    12: 1.41,  # Mg
    13: 1.21,  # Al
    14: 1.11,  # Si
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
    18: 1.06,  # Ar
    19: 2.03,  # K
    20: 1.76,  # Ca
    21: 1.70,  # Sc
    22: 1.60,  # Ti
    23: 1.53,  # V
    24: 1.39,  # Cr
    25: 1.39,  # Mn
    26: 1.32,  # Fe
    27: 1.26,  # Co
    28: 1.24,  # Ni
    29: 1.32,  # Cu
    30: 1.22,  # Zn
    31: 1.22,  # Ga
    32: 1.20,  # Ge
    33: 1.19,  # As
    34: 1.20,  # Se
    35: 1.20,  # Br
    36: 1.16,  # Kr
    37: 2.20,  # Rb
    38: 1.95,  # Sr
    39: 1.90,  # Y
    40: 1.75,  # Zr
    41: 1.64,  # Nb
    42: 1.54,  # Mo
    44: 1.46,  # Ru
    45: 1.42,  # Rh
    46: 1.39,  # Pd
    47: 1.45,  # Ag
    48: 1.44,  # Cd
    49: 1.42,  # In
    50: 1.39,  # Sn
    51: 1.39,  # Sb
    52: 1.38,  # Te
    53: 1.39,  # I
    54: 1.40,  # Xe
}

# 이온화 에너지 (eV, first ionization)
IONIZATION_ENERGY: Final[Dict[int, float]] = {
    1: 13.598,   # H
    2: 24.587,   # He
    3: 5.392,    # Li
    4: 9.323,    # Be
    5: 8.298,    # B
    6: 11.260,   # C
    7: 14.534,   # N
    8: 13.618,   # O
    9: 17.423,   # F
    10: 21.565,  # Ne
    11: 5.139,   # Na
    12: 7.646,   # Mg
    13: 5.986,   # Al
    14: 8.152,   # Si
    15: 10.487,  # P
    16: 10.360,  # S
    17: 12.968,  # Cl
    18: 15.760,  # Ar
    19: 4.341,   # K
    20: 6.113,   # Ca
    21: 6.561,   # Sc
    22: 6.828,   # Ti
    23: 6.746,   # V
    24: 6.767,   # Cr
    25: 7.434,   # Mn
    26: 7.902,   # Fe
    27: 7.881,   # Co
    28: 7.640,   # Ni
    29: 7.726,   # Cu
    30: 9.394,   # Zn
    31: 5.999,   # Ga
    32: 7.899,   # Ge
    33: 9.789,   # As
    34: 9.752,   # Se
    35: 11.814,  # Br
    36: 14.000,  # Kr
}

# 원소 기호
ELEMENT_SYMBOLS: Final[Dict[int, str]] = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
    23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
    30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi',
}

# 역방향 매핑: 기호 -> 원자번호
ATOMIC_NUMBERS: Final[Dict[str, int]] = {v: k for k, v in ELEMENT_SYMBOLS.items()}

# 최대 결합 길이 (Angstrom)
MAX_BOND_LENGTH: Final[float] = 3.5

# 기본 결합 길이 (Angstrom, 원소쌍별)
DEFAULT_BOND_LENGTHS: Final[Dict[str, float]] = {
    'C-C': 1.54, 'C=C': 1.34, 'C≡C': 1.20,
    'C-H': 1.09, 'C-N': 1.47, 'C=N': 1.29,
    'C-O': 1.43, 'C=O': 1.23, 'N-H': 1.01,
    'O-H': 0.96, 'Si-O': 1.63, 'Si-Si': 2.35,
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