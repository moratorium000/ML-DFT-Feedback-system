"""Tests for models.validation module."""
import pytest
import numpy as np

from models.validation.metrics import (
    MetricsCalculator,
    PropertyMetrics,
    StructuralMetrics
)


class TestMetricsCalculator:
    """Test MetricsCalculator class."""

    def test_calculator_initialization(self):
        """Test calculator can be instantiated."""
        calculator = MetricsCalculator()
        assert calculator is not None

    def test_mae_calculation(self):
        """Test mean absolute error calculation."""
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.1, 2.2, 2.8])

        calculator = MetricsCalculator()
        mae = calculator.calculate_mae(predicted, actual)

        expected_mae = np.mean(np.abs(predicted - actual))
        np.testing.assert_almost_equal(mae, expected_mae)

    def test_rmse_calculation(self):
        """Test root mean square error calculation."""
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 3.0])

        calculator = MetricsCalculator()
        rmse = calculator.calculate_rmse(predicted, actual)

        np.testing.assert_almost_equal(rmse, 0.0)

    def test_r2_calculation(self):
        """Test R-squared calculation."""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        calculator = MetricsCalculator()
        r2 = calculator.calculate_r2(predicted, actual)

        np.testing.assert_almost_equal(r2, 1.0)


class TestPropertyMetrics:
    """Test PropertyMetrics dataclass."""

    def test_property_metrics_creation(self):
        """Test property metrics can be created."""
        metrics = PropertyMetrics(
            property_name="band_gap",
            predicted_value=1.5,
            target_value=1.2,
            absolute_error=0.3,
            relative_error=0.25
        )

        assert metrics.property_name == "band_gap"
        assert metrics.predicted_value == 1.5
        assert metrics.target_value == 1.2


class TestStructuralMetrics:
    """Test StructuralMetrics dataclass."""

    def test_structural_metrics_creation(self):
        """Test structural metrics can be created."""
        metrics = StructuralMetrics(
            rmsd=0.5,
            max_displacement=1.2,
            volume_change=0.05,
            bond_length_changes={"Si-Si": 0.02}
        )

        assert metrics.rmsd == 0.5
        assert metrics.max_displacement == 1.2


class TestValidationChecker:
    """Test validation checker functionality."""

    def test_valid_structure(self, sample_structure):
        """Test validation of valid structure."""
        from models.validation.checker import StructureChecker

        checker = StructureChecker()
        result = checker.check_geometry(sample_structure)

        assert hasattr(result, 'is_valid')

    def test_distance_check(self, sample_structure):
        """Test atomic distance validation."""
        from models.validation.checker import StructureChecker

        checker = StructureChecker()
        result = checker.check_distances(sample_structure)

        assert isinstance(result, dict)

    def test_angle_check(self, sample_structure):
        """Test bond angle validation."""
        from models.validation.checker import StructureChecker

        checker = StructureChecker()
        result = checker.check_angles(sample_structure)

        assert isinstance(result, dict)
