"""Tests for core.interfaces module."""
import pytest
import numpy as np
from datetime import datetime

from core.interfaces import (
    Structure,
    MutationResult,
    PredictionResult,
    DFTResult,
    ValidationResult,
    CalculationStatus
)


class TestStructure:
    """Test Structure dataclass."""

    def test_structure_creation(self, sample_lattice, sample_positions, sample_atomic_numbers):
        """Test basic structure creation."""
        structure = Structure(
            lattice_vectors=sample_lattice,
            positions=sample_positions,
            atomic_numbers=np.array(sample_atomic_numbers),
            cell_params={"a": 5.0, "b": 5.0, "c": 5.0},
            formula="Si4"
        )

        assert structure.lattice_vectors.shape == (3, 3)
        assert structure.positions.shape == (4, 3)
        assert len(structure.atomic_numbers) == 4
        assert structure.formula == "Si4"

    def test_structure_with_space_group(self, sample_structure):
        """Test structure with space group."""
        structure = Structure(
            lattice_vectors=sample_structure.lattice_vectors,
            positions=sample_structure.positions,
            atomic_numbers=sample_structure.atomic_numbers,
            cell_params=sample_structure.cell_params,
            formula=sample_structure.formula,
            space_group="Fm-3m"
        )
        assert structure.space_group == "Fm-3m"

    def test_structure_properties(self, sample_structure):
        """Test structure has expected properties."""
        assert hasattr(sample_structure, 'lattice_vectors')
        assert hasattr(sample_structure, 'positions')
        assert hasattr(sample_structure, 'atomic_numbers')
        assert hasattr(sample_structure, 'formula')
        assert hasattr(sample_structure, 'cell_params')


class TestMutationResult:
    """Test MutationResult dataclass."""

    def test_mutation_result_creation(self, sample_structure):
        """Test mutation result creation."""
        result = MutationResult(
            original_structure=sample_structure,
            mutated_structure=sample_structure,
            mutation_type="substitution",
            changes={"element": "Ge", "site": 0},
            success=True,
            stability_score=0.9,
            validity_score=0.85,
            energy_estimate=-100.0,
            generation=1,
            parent_id=None
        )

        assert result.mutation_type == "substitution"
        assert result.changes["element"] == "Ge"

    def test_mutation_result_with_scores(self, sample_structure):
        """Test mutation result with scores."""
        result = MutationResult(
            original_structure=sample_structure,
            mutated_structure=sample_structure,
            mutation_type="distortion",
            changes={"magnitude": 0.1},
            success=True,
            stability_score=0.9,
            validity_score=0.85,
            energy_estimate=-100.0,
            generation=1,
            parent_id="parent_001"
        )

        assert result.success is True
        assert result.stability_score == 0.9
        assert result.validity_score == 0.85


class TestDFTResult:
    """Test DFTResult dataclass."""

    def test_dft_result_creation(self, sample_structure):
        """Test DFT result creation."""
        result = DFTResult(
            initial_structure=sample_structure,
            final_structure=sample_structure,
            total_energy=-100.5,
            energy_per_atom=-25.125,
            formation_energy=-0.5,
            forces=np.zeros((4, 3)),
            stress_tensor=np.zeros((3, 3)),
            band_gap=1.5,
            dos=None,
            band_structure=None,
            convergence=True,
            calculation_time=100.0,
            error_messages=[]
        )

        assert result.total_energy == -100.5
        assert result.convergence is True

    def test_dft_result_with_forces(self, sample_structure):
        """Test DFT result with forces."""
        forces = np.random.randn(4, 3)
        result = DFTResult(
            initial_structure=sample_structure,
            final_structure=sample_structure,
            total_energy=-100.5,
            energy_per_atom=-25.125,
            formation_energy=-0.5,
            forces=forces,
            stress_tensor=np.zeros((3, 3)),
            band_gap=None,
            dos=None,
            band_structure=None,
            convergence=True,
            calculation_time=100.0,
            error_messages=[]
        )

        assert result.forces is not None
        assert result.forces.shape == (4, 3)

    def test_dft_result_with_electronic_properties(self, sample_structure):
        """Test DFT result with electronic properties."""
        result = DFTResult(
            initial_structure=sample_structure,
            final_structure=sample_structure,
            total_energy=-100.5,
            energy_per_atom=-25.125,
            formation_energy=-0.5,
            forces=np.zeros((4, 3)),
            stress_tensor=np.zeros((3, 3)),
            band_gap=1.5,
            dos={"total": [1, 2, 3]},
            band_structure={"bands": [[1, 2], [3, 4]]},
            convergence=True,
            calculation_time=100.0,
            error_messages=[]
        )

        assert result.band_gap == 1.5
        assert result.dos is not None


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ValidationResult(
            is_valid=True,
            stability_score=0.95,
            validation_details={"geometry": "ok", "chemistry": "ok"},
            error_messages=[]
        )

        assert result.is_valid is True
        assert result.stability_score == 0.95
        assert len(result.error_messages) == 0

    def test_validation_result_with_errors(self):
        """Test validation result with errors."""
        result = ValidationResult(
            is_valid=False,
            stability_score=0.3,
            validation_details={"geometry": "failed"},
            error_messages=["Atomic distance too small", "Invalid cell angle"]
        )

        assert result.is_valid is False
        assert len(result.error_messages) == 2


class TestCalculationStatus:
    """Test CalculationStatus enum."""

    def test_status_values(self):
        """Test calculation status enum values."""
        assert CalculationStatus.PENDING is not None
        assert CalculationStatus.RUNNING is not None
        assert CalculationStatus.COMPLETED is not None
        assert CalculationStatus.FAILED is not None

    def test_status_string_values(self):
        """Test calculation status string values."""
        assert CalculationStatus.PENDING.value == "pending"
        assert CalculationStatus.COMPLETED.value == "completed"
