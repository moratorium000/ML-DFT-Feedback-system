"""pytest fixtures for ML-DFT Feedback System tests."""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_lattice():
    """Sample cubic lattice vectors (5 Angstrom)."""
    return np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ])


@pytest.fixture
def sample_positions():
    """Sample fractional positions for 4 atoms."""
    return np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])


@pytest.fixture
def sample_atomic_numbers():
    """Sample atomic numbers (Si-like structure)."""
    return [14, 14, 14, 14]


@pytest.fixture
def sample_structure(sample_lattice, sample_positions, sample_atomic_numbers):
    """Sample Structure object."""
    from core.interfaces import Structure
    return Structure(
        lattice_vectors=sample_lattice,
        positions=sample_positions,
        atomic_numbers=np.array(sample_atomic_numbers),
        cell_params={"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
        formula="Si4"
    )


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_json_file(temp_dir):
    """Create a sample JSON file."""
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    file_path = temp_dir / "test.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def sample_yaml_file(temp_dir):
    """Create a sample YAML file."""
    import yaml
    data = {"config": {"param1": 1, "param2": "value"}}
    file_path = temp_dir / "test.yaml"
    with open(file_path, 'w') as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
def sample_dft_result(sample_structure):
    """Sample DFT calculation result."""
    from core.interfaces import DFTResult
    return DFTResult(
        initial_structure=sample_structure,
        final_structure=sample_structure,
        total_energy=-100.5,
        energy_per_atom=-25.125,
        formation_energy=-0.5,
        forces=np.random.randn(4, 3) * 0.1,
        stress_tensor=np.random.randn(3, 3) * 0.01,
        band_gap=1.5,
        dos=None,
        band_structure=None,
        convergence=True,
        calculation_time=120.5,
        error_messages=[]
    )


@pytest.fixture
def sample_target_properties():
    """Sample target properties for optimization."""
    return {
        "band_gap": 1.2,
        "formation_energy": -0.5,
        "density": 2.33
    }
