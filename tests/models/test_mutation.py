"""Tests for models.mutation module."""
import pytest
import numpy as np

from models.mutation.generator import MutationGenerator, MutationType
from models.mutation.validator import MutationValidator


class TestMutationGenerator:
    """Test MutationGenerator class."""

    def test_generator_initialization(self):
        """Test generator can be instantiated."""
        generator = MutationGenerator()
        assert generator is not None

    def test_substitution_mutation(self, sample_structure):
        """Test elemental substitution mutation."""
        generator = MutationGenerator()
        result = generator.generate_mutation(
            structure=sample_structure,
            mutation_type=MutationType.SUBSTITUTION
        )

        assert result is not None
        assert result.mutation_type == "substitution"
        assert result.mutated_structure is not None

    def test_distortion_mutation(self, sample_structure):
        """Test structural distortion mutation."""
        generator = MutationGenerator()
        result = generator.generate_mutation(
            structure=sample_structure,
            mutation_type=MutationType.DISTORTION
        )

        assert result is not None
        assert result.mutation_type == "distortion"

    def test_vacancy_mutation(self, sample_structure):
        """Test vacancy creation mutation."""
        generator = MutationGenerator()
        result = generator.generate_mutation(
            structure=sample_structure,
            mutation_type=MutationType.VACANCY
        )

        assert result is not None
        assert result.mutation_type == "vacancy"
        # Should have fewer atoms
        assert len(result.mutated_structure.positions) < len(sample_structure.positions)

    def test_multiple_mutations(self, sample_structure):
        """Test generating multiple mutations."""
        generator = MutationGenerator()
        results = generator.generate_mutations(
            structure=sample_structure,
            n_mutations=5
        )

        assert len(results) == 5
        for result in results:
            assert result.mutated_structure is not None


class TestMutationValidator:
    """Test MutationValidator class."""

    def test_validator_initialization(self):
        """Test validator can be instantiated."""
        validator = MutationValidator()
        assert validator is not None

    def test_validate_mutation(self, sample_structure):
        """Test mutation validation."""
        validator = MutationValidator()

        # Create a simple mutation result
        from core.interfaces import MutationResult
        mutation = MutationResult(
            original_structure=sample_structure,
            mutated_structure=sample_structure,
            mutation_type="substitution",
            mutation_params={}
        )

        result = validator.validate(mutation)
        assert hasattr(result, 'is_valid')

    def test_validate_structure_constraints(self, sample_structure):
        """Test structure constraint validation."""
        validator = MutationValidator()
        result = validator.check_structure_constraints(sample_structure)

        assert isinstance(result, dict)
        assert 'distance_valid' in result or 'is_valid' in result

    def test_validate_chemical_feasibility(self, sample_structure):
        """Test chemical feasibility validation."""
        validator = MutationValidator()
        result = validator.check_chemical_feasibility(sample_structure)

        assert isinstance(result, dict)


class TestMutationType:
    """Test MutationType enum."""

    def test_mutation_types_exist(self):
        """Test that expected mutation types exist."""
        assert MutationType.SUBSTITUTION is not None
        assert MutationType.DISTORTION is not None
        assert MutationType.VACANCY is not None

    def test_mutation_type_values(self):
        """Test mutation type string values."""
        assert MutationType.SUBSTITUTION.value == "substitution"
        assert MutationType.DISTORTION.value == "distortion"
        assert MutationType.VACANCY.value == "vacancy"
