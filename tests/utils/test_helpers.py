"""Tests for utils.helpers module."""
import pytest
import numpy as np
import json
import yaml
from pathlib import Path

from utils.helpers import (
    generate_unique_id,
    calculate_distances,
    calculate_angles,
    calculate_volume,
    convert_to_cartesian,
    convert_to_fractional,
    load_json_file,
    save_json_file,
    load_yaml_file,
    save_yaml_file,
    format_time,
    format_size,
    deep_update
)


class TestGenerateUniqueId:
    """Test generate_unique_id function."""

    def test_generates_string(self):
        """Test that function returns a string."""
        result = generate_unique_id()
        assert isinstance(result, str)

    def test_with_prefix(self):
        """Test with prefix parameter."""
        result = generate_unique_id(prefix="test")
        assert result.startswith("test_")

    def test_unique_ids(self):
        """Test that generated IDs are unique."""
        ids = [generate_unique_id() for _ in range(100)]
        assert len(ids) == len(set(ids))


class TestCalculateDistances:
    """Test calculate_distances function."""

    def test_cubic_lattice(self, sample_lattice, sample_positions):
        """Test distance calculation in cubic lattice."""
        distances = calculate_distances(sample_positions, sample_lattice)

        assert distances.shape == (4, 4)
        # Diagonal should be zero
        np.testing.assert_array_equal(np.diag(distances), 0)
        # Should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)

    def test_self_distance_zero(self, sample_lattice):
        """Test that self-distance is zero."""
        positions = np.array([[0.0, 0.0, 0.0]])
        distances = calculate_distances(positions, sample_lattice)
        assert distances[0, 0] == 0.0

    def test_known_distance(self):
        """Test with known distance."""
        lattice = np.eye(3) * 10.0  # 10 Angstrom cubic
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0]  # 5 Angstrom apart
        ])
        distances = calculate_distances(positions, lattice)
        np.testing.assert_almost_equal(distances[0, 1], 5.0)


class TestCalculateAngles:
    """Test calculate_angles function."""

    def test_right_angle(self):
        """Test calculation of 90 degree angle."""
        positions = np.array([
            [0.0, 0.0, 0.0],  # center
            [1.0, 0.0, 0.0],  # x-direction
            [0.0, 1.0, 0.0]   # y-direction
        ])
        angles = calculate_angles(positions, center_idx=0, neighbor_indices=[1, 2])
        np.testing.assert_almost_equal(angles[0], 90.0)

    def test_linear_angle(self):
        """Test calculation of 180 degree angle."""
        positions = np.array([
            [0.0, 0.0, 0.0],  # center
            [1.0, 0.0, 0.0],  # +x
            [-1.0, 0.0, 0.0]  # -x
        ])
        angles = calculate_angles(positions, center_idx=0, neighbor_indices=[1, 2])
        np.testing.assert_almost_equal(angles[0], 180.0)

    def test_multiple_angles(self):
        """Test with multiple neighbors."""
        positions = np.array([
            [0.0, 0.0, 0.0],  # center
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        angles = calculate_angles(positions, center_idx=0, neighbor_indices=[1, 2, 3])
        # Should have 3 angles (C(3,2) = 3)
        assert len(angles) == 3
        # All should be 90 degrees
        np.testing.assert_array_almost_equal(angles, [90.0, 90.0, 90.0])


class TestCalculateVolume:
    """Test calculate_volume function."""

    def test_cubic_volume(self, sample_lattice):
        """Test volume of cubic cell."""
        volume = calculate_volume(sample_lattice)
        np.testing.assert_almost_equal(volume, 125.0)  # 5^3

    def test_unit_cube(self):
        """Test volume of unit cube."""
        lattice = np.eye(3)
        volume = calculate_volume(lattice)
        np.testing.assert_almost_equal(volume, 1.0)

    def test_scaled_cube(self):
        """Test volume scales correctly."""
        lattice = np.eye(3) * 2.0
        volume = calculate_volume(lattice)
        np.testing.assert_almost_equal(volume, 8.0)


class TestCoordinateConversion:
    """Test coordinate conversion functions."""

    def test_cartesian_to_fractional_roundtrip(self, sample_lattice, sample_positions):
        """Test conversion roundtrip."""
        cartesian = convert_to_cartesian(sample_positions, sample_lattice)
        fractional = convert_to_fractional(cartesian, sample_lattice)
        np.testing.assert_array_almost_equal(fractional, sample_positions)

    def test_cartesian_conversion(self):
        """Test fractional to cartesian conversion."""
        lattice = np.eye(3) * 10.0
        positions = np.array([[0.5, 0.5, 0.5]])
        cartesian = convert_to_cartesian(positions, lattice)
        np.testing.assert_array_almost_equal(cartesian, [[5.0, 5.0, 5.0]])


class TestFileIO:
    """Test file I/O functions."""

    def test_json_roundtrip(self, temp_dir):
        """Test JSON save and load."""
        data = {"key": "value", "number": 42}
        file_path = temp_dir / "test.json"

        save_json_file(data, file_path)
        loaded = load_json_file(file_path)

        assert loaded == data

    def test_yaml_roundtrip(self, temp_dir):
        """Test YAML save and load."""
        data = {"config": {"param": 1}}
        file_path = temp_dir / "test.yaml"

        save_yaml_file(data, file_path)
        loaded = load_yaml_file(file_path)

        assert loaded == data

    def test_load_json_file(self, sample_json_file):
        """Test loading existing JSON file."""
        data = load_json_file(sample_json_file)
        assert data["key"] == "value"
        assert data["number"] == 42

    def test_load_yaml_file(self, sample_yaml_file):
        """Test loading existing YAML file."""
        data = load_yaml_file(sample_yaml_file)
        assert data["config"]["param1"] == 1


class TestFormatTime:
    """Test format_time function."""

    def test_seconds(self):
        """Test formatting seconds."""
        assert format_time(30.5) == "30.5s"

    def test_minutes(self):
        """Test formatting minutes."""
        assert format_time(90) == "1.5m"

    def test_hours(self):
        """Test formatting hours."""
        assert format_time(3600) == "1.0h"

    def test_multiple_hours(self):
        """Test formatting multiple hours."""
        assert format_time(7200) == "2.0h"


class TestFormatSize:
    """Test format_size function."""

    def test_bytes(self):
        """Test formatting bytes."""
        assert format_size(500) == "500.0B"

    def test_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_size(2048) == "2.0KB"

    def test_megabytes(self):
        """Test formatting megabytes."""
        assert format_size(2 * 1024 * 1024) == "2.0MB"


class TestDeepUpdate:
    """Test deep_update function."""

    def test_simple_update(self):
        """Test simple dictionary update."""
        d1 = {"a": 1}
        d2 = {"b": 2}
        result = deep_update(d1, d2)
        assert result == {"a": 1, "b": 2}

    def test_nested_update(self):
        """Test nested dictionary update."""
        d1 = {"a": {"b": 1, "c": 2}}
        d2 = {"a": {"b": 10}}
        result = deep_update(d1, d2)
        assert result == {"a": {"b": 10, "c": 2}}

    def test_deep_nested_update(self):
        """Test deeply nested update."""
        d1 = {"a": {"b": {"c": 1}}}
        d2 = {"a": {"b": {"d": 2}}}
        result = deep_update(d1, d2)
        assert result == {"a": {"b": {"c": 1, "d": 2}}}
