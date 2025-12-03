"""Tests for data.cache module."""
import pytest
import numpy as np
from pathlib import Path

from data.cache.memory import MemoryCache
from data.cache.disk import DiskCache, DiskCacheConfig


class TestMemoryCache:
    """Test MemoryCache class."""

    def test_cache_initialization(self):
        """Test cache can be instantiated."""
        cache = MemoryCache(max_size=100)
        assert cache is not None

    def test_set_and_get(self):
        """Test setting and getting values."""
        cache = MemoryCache(max_size=100)
        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

    def test_get_missing_key(self):
        """Test getting non-existent key."""
        cache = MemoryCache(max_size=100)
        result = cache.get("missing_key")
        assert result is None

    def test_delete(self):
        """Test deleting key."""
        cache = MemoryCache(max_size=100)
        cache.set("key1", "value1")
        cache.delete("key1")
        result = cache.get("key1")
        assert result is None

    def test_clear(self):
        """Test clearing cache."""
        cache = MemoryCache(max_size=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_max_size_eviction(self):
        """Test LRU eviction when max size reached."""
        cache = MemoryCache(max_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_store_numpy_array(self):
        """Test storing numpy arrays."""
        cache = MemoryCache(max_size=100)
        arr = np.array([1, 2, 3, 4, 5])
        cache.set("array", arr)
        result = cache.get("array")
        np.testing.assert_array_equal(result, arr)

    def test_store_dict(self):
        """Test storing dictionaries."""
        cache = MemoryCache(max_size=100)
        data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        cache.set("dict", data)
        result = cache.get("dict")
        assert result == data


class TestDiskCache:
    """Test DiskCache class."""

    @pytest.fixture
    def disk_cache(self, temp_dir):
        """Create a disk cache for testing."""
        config = DiskCacheConfig(
            cache_dir=temp_dir,
            max_size_bytes=1024 * 1024,
            compression=False
        )
        return DiskCache(config)

    def test_cache_initialization(self, disk_cache):
        """Test cache can be instantiated."""
        assert disk_cache is not None

    @pytest.mark.asyncio
    async def test_async_set_and_get(self, disk_cache):
        """Test async setting and getting values."""
        await disk_cache.set("key1", {"data": "value1"})
        result = await disk_cache.get("key1")
        assert result["data"] == "value1"

    @pytest.mark.asyncio
    async def test_async_get_missing_key(self, disk_cache):
        """Test getting non-existent key."""
        result = await disk_cache.get("missing_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_async_delete(self, disk_cache):
        """Test deleting key."""
        await disk_cache.set("key1", {"data": "value1"})
        await disk_cache.delete("key1")
        result = await disk_cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_complex_data(self, disk_cache):
        """Test storing complex data structures."""
        data = {
            "array": [1, 2, 3, 4, 5],
            "nested": {"a": 1, "b": 2},
            "string": "test value"
        }
        await disk_cache.set("complex", data)
        result = await disk_cache.get("complex")
        assert result == data


class TestCacheManager:
    """Test CacheManager class."""

    def test_manager_initialization(self, temp_dir):
        """Test cache manager can be instantiated."""
        from data.cache.manager import CacheManager

        manager = CacheManager(
            cache_dir=temp_dir,
            memory_max_size=100
        )
        assert manager is not None

    def test_get_from_empty_cache(self, temp_dir):
        """Test getting from empty cache."""
        from data.cache.manager import CacheManager

        manager = CacheManager(
            cache_dir=temp_dir,
            memory_max_size=100
        )
        result = manager.get("missing")
        assert result is None
