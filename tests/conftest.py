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
        atomic_numbers=sample_atomic_numbers,
        species=["Si", "Si", "Si", "Si"]
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
def sample_dft_result():
    """Sample DFT calculation result."""
    from core.interfaces import DFTResult, CalculationStatus
    return DFTResult(
        structure_id="test_001",
        total_energy=-100.5,
        forces=np.random.randn(4, 3) * 0.1,
        stress=np.random.randn(3, 3) * 0.01,
        band_gap=1.5,
        fermi_energy=-3.2,
        status=CalculationStatus.COMPLETED
    )


@pytest.fixture
def sample_target_properties():
    """Sample target properties for optimization."""
    return {
        "band_gap": 1.2,
        "formation_energy": -0.5,
        "density": 2.33
    }
