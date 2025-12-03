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
    CalculationStatus,
    OptimizationStatus
)


class TestStructure:
    """Test Structure dataclass."""

    def test_structure_creation(self, sample_lattice, sample_positions, sample_atomic_numbers):
        """Test basic structure creation."""
        structure = Structure(
            lattice_vectors=sample_lattice,
            positions=sample_positions,
            atomic_numbers=sample_atomic_numbers,
            species=["Si"] * 4
        )

        assert structure.lattice_vectors.shape == (3, 3)
        assert structure.positions.shape == (4, 3)
        assert len(structure.atomic_numbers) == 4
        assert len(structure.species) == 4

    def test_structure_with_id(self, sample_structure):
        """Test structure with custom ID."""
        structure = Structure(
            lattice_vectors=sample_structure.lattice_vectors,
            positions=sample_structure.positions,
            atomic_numbers=sample_structure.atomic_numbers,
            species=sample_structure.species,
            structure_id="custom_id_001"
        )
        assert structure.structure_id == "custom_id_001"

    def test_structure_properties(self, sample_structure):
        """Test structure has expected properties."""
        assert hasattr(sample_structure, 'lattice_vectors')
        assert hasattr(sample_structure, 'positions')
        assert hasattr(sample_structure, 'atomic_numbers')
        assert hasattr(sample_structure, 'species')


class TestMutationResult:
    """Test MutationResult dataclass."""

    def test_mutation_result_creation(self, sample_structure):
        """Test mutation result creation."""
        result = MutationResult(
            original_structure=sample_structure,
            mutated_structure=sample_structure,
            mutation_type="substitution",
            mutation_params={"element": "Ge", "site": 0}
        )

        assert result.mutation_type == "substitution"
        assert result.mutation_params["element"] == "Ge"

    def test_mutation_result_with_score(self, sample_structure):
        """Test mutation result with score."""
        result = MutationResult(
            original_structure=sample_structure,
            mutated_structure=sample_structure,
            mutation_type="distortion",
            mutation_params={"magnitude": 0.1},
            success=True,
            score=0.85
        )

        assert result.success is True
        assert result.score == 0.85


class TestDFTResult:
    """Test DFTResult dataclass."""

    def test_dft_result_creation(self):
        """Test DFT result creation."""
        result = DFTResult(
            structure_id="test_001",
            total_energy=-100.5,
            status=CalculationStatus.COMPLETED
        )

        assert result.total_energy == -100.5
        assert result.status == CalculationStatus.COMPLETED

    def test_dft_result_with_forces(self):
        """Test DFT result with forces."""
        forces = np.random.randn(4, 3)
        result = DFTResult(
            structure_id="test_002",
            total_energy=-100.5,
            forces=forces,
            status=CalculationStatus.COMPLETED
        )

        assert result.forces is not None
        assert result.forces.shape == (4, 3)

    def test_dft_result_with_electronic_properties(self):
        """Test DFT result with electronic properties."""
        result = DFTResult(
            structure_id="test_003",
            total_energy=-100.5,
            band_gap=1.5,
            fermi_energy=-3.2,
            status=CalculationStatus.COMPLETED
        )

        assert result.band_gap == 1.5
        assert result.fermi_energy == -3.2


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor overlap detected"]
        )

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1

    def test_validation_result_with_errors(self):
        """Test validation result with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Atomic distance too small", "Invalid cell angle"],
            warnings=[]
        )

        assert result.is_valid is False
        assert len(result.errors) == 2


class TestCalculationStatus:
    """Test CalculationStatus enum."""

    def test_status_values(self):
        """Test calculation status enum values."""
        assert CalculationStatus.PENDING is not None
        assert CalculationStatus.RUNNING is not None
        assert CalculationStatus.COMPLETED is not None
        assert CalculationStatus.FAILED is not None


class TestOptimizationStatus:
    """Test OptimizationStatus enum."""

    def test_status_values(self):
        """Test optimization status enum values."""
        assert OptimizationStatus.INITIALIZING is not None
        assert OptimizationStatus.RUNNING is not None
        assert OptimizationStatus.CONVERGED is not None
        assert OptimizationStatus.FAILED is not None
