from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
import json
import yaml
from datetime import *
import hashlib
import uuid
from utils.constants import *
from core.interfaces import Structure


def generate_unique_id(prefix: str = "") -> str:
    """고유 ID 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{random_suffix}"


def calculate_distances(positions: np.ndarray,
                        lattice: np.ndarray) -> np.ndarray:
    """원자간 거리 계산"""
    n_atoms = len(positions)
    distances = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            diff = positions[i] - positions[j]
            diff = diff - np.round(diff)  # 최소 이미지 규약
            cart_diff = np.dot(diff, lattice)
            dist = np.linalg.norm(cart_diff)
            distances[i, j] = distances[j, i] = dist

    return distances


def calculate_angles(positions: np.ndarray,
                     center_idx: int,
                     neighbor_indices: List[int]) -> np.ndarray:
    """결합각 계산"""
    angles = []
    center = positions[center_idx]

    for i in range(len(neighbor_indices)):
        for j in range(i + 1, len(neighbor_indices)):
            vec1 = positions[neighbor_indices[i]] - center
            vec2 = positions[neighbor_indices[j]] - center

            # 각도 계산 (라디안)
            cos_angle = np.dot(vec1, vec2) / (
                    np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

            # 도(degree)로 변환
            angles.append(np.degrees(angle))

    return np.array(angles)


def find_neighbors(structure: Structure,
                   center_idx: int,
                   max_distance: float = MAX_ATOMIC_DISTANCE) -> List[int]:
    """주변 원자 찾기"""
    positions = structure.positions
    lattice = structure.lattice_vectors
    neighbors = []

    for i in range(len(positions)):
        if i == center_idx:
            continue

        diff = positions[i] - positions[center_idx]
        diff = diff - np.round(diff)
        cart_diff = np.dot(diff, lattice)
        distance = np.linalg.norm(cart_diff)

        if distance <= max_distance:
            neighbors.append(i)

    return neighbors


def calculate_volume(lattice: np.ndarray) -> float:
    """단위 셀 부피 계산"""
    return abs(np.linalg.det(lattice))


def calculate_density(structure: Structure) -> float:
    """밀도 계산"""
    volume = calculate_volume(structure.lattice_vectors)
    total_mass = sum(ATOMIC_MASS[z] for z in structure.atomic_numbers)
    return total_mass / volume


def convert_to_cartesian(positions: np.ndarray,
                         lattice: np.ndarray) -> np.ndarray:
    """분율 좌표를 직교 좌표로 변환"""
    return np.dot(positions, lattice)


def convert_to_fractional(positions: np.ndarray,
                          lattice: np.ndarray) -> np.ndarray:
    """직교 좌표를 분율 좌표로 변환"""
    inv_lattice = np.linalg.inv(lattice)
    return np.dot(positions, inv_lattice)


def load_json_file(file_path: Union[str, Path]) -> Dict:
    """JSON 파일 로드"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json_file(data: Dict,
                   file_path: Union[str, Path],
                   indent: int = 2):
    """JSON 파일 저장"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_yaml_file(file_path: Union[str, Path]) -> Dict:
    """YAML 파일 로드"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml_file(data: Dict,
                   file_path: Union[str, Path]):
    """YAML 파일 저장"""
    with open(file_path, 'w') as f:
        yaml.dump(data, f)


def calculate_rmsd(struct1: Structure,
                   struct2: Structure) -> float:
    """RMSD 계산"""
    if len(struct1.positions) != len(struct2.positions):
        raise ValueError("Structures have different numbers of atoms")

    pos1 = convert_to_cartesian(struct1.positions, struct1.lattice_vectors)
    pos2 = convert_to_cartesian(struct2.positions, struct2.lattice_vectors)

    return np.sqrt(np.mean(np.sum((pos1 - pos2) ** 2, axis=1)))


def validate_structure(structure: Structure) -> Tuple[bool, List[str]]:
    """구조 유효성 검사"""
    errors = []

    # 원자간 거리 검사
    distances = calculate_distances(
        structure.positions,
        structure.lattice_vectors
    )
    min_dist = np.min(distances[distances > 0])
    if min_dist < MIN_ATOMIC_DISTANCE:
        errors.append(f"Atomic distance {min_dist:.2f} Å is too small")

    # 격자 각도 검사
    for angle in structure.cell_params.values():
        if not (MIN_CELL_ANGLE <= angle <= MAX_CELL_ANGLE):
            errors.append(f"Cell angle {angle:.2f}° is out of range")

    # 부피 검사
    volume = calculate_volume(structure.lattice_vectors)
    if volume <= 0:
        errors.append("Cell volume is negative or zero")

    return len(errors) == 0, errors


def format_time(seconds: float) -> str:
    """시간 형식화"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_size(bytes: int) -> str:
    """크기 형식화"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024
    return f"{bytes:.1f}TB"


def deep_update(d1: Dict, d2: Dict) -> Dict:
    """딕셔너리 재귀적 업데이트"""
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            deep_update(d1[k], v)
        else:
            d1[k] = v
    return d1