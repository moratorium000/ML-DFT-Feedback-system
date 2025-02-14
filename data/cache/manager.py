from typing import Dict, Optional, Any, Union
from pathlib import Path
import logging
from memory import MemoryCache
from disk import DiskCache, DiskCacheConfig


class CacheManager:
    """캐시 관리 시스템"""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)

        # 메모리 캐시 초기화
        self.structure_cache = MemoryCache(
            max_size=config.get('structure_cache_size', 500),
            ttl=config.get('structure_cache_ttl', 7200)
        )

        self.calculation_cache = MemoryCache(
            max_size=config.get('calculation_cache_size', 200),
            ttl=config.get('calculation_cache_ttl', 3600)
        )

        self.prediction_cache = MemoryCache(
            max_size=config.get('prediction_cache_size', 300),
            ttl=config.get('prediction_cache_ttl', 1800)
        )

        # 디스크 캐시 초기화
        disk_config = DiskCacheConfig(
            base_path=Path(config.get('cache_dir', '.cache')),
            max_size_gb=config.get('max_cache_size_gb', 10.0),
            compression=config.get('compression', True)
        )
        self.disk_cache = DiskCache(disk_config)

    async def get_structure(self, structure_id: str) -> Optional[Dict]:
        """구조 데이터 조회"""
        # 메모리 캐시 확인
        data = self.structure_cache.get(structure_id)
        if data is not None:
            return data

        # 디스크 캐시 확인
        data = await self.disk_cache.get(structure_id)
        if data is not None:
            # 메모리 캐시에 저장
            self.structure_cache.set(structure_id, data)
            return data

        return None

    async def set_structure(self,
                            structure_id: str,
                            data: Dict,
                            ttl: Optional[int] = None) -> bool:
        """구조 데이터 저장"""
        try:
            # 메모리 캐시에 저장
            self.structure_cache.set(structure_id, data, ttl)

            # 디스크 캐시에 저장
            await self.disk_cache.set(structure_id, data, ttl)

            return True
        except Exception as e:
            self.logger.error(f"Error caching structure {structure_id}: {e}")
            return False

    async def get_calculation(self, calc_id: str) -> Optional[Dict]:
        """계산 결과 조회"""
        # 메모리 캐시 확인
        data = self.calculation_cache.get(calc_id)
        if data is not None:
            return data

        # 디스크 캐시 확인
        data = await self.disk_cache.get(f"calc_{calc_id}")
        if data is not None:
            # 메모리 캐시에 저장
            self.calculation_cache.set(calc_id, data)
            return data

        return None

    async def set_calculation(self,
                              calc_id: str,
                              data: Dict,
                              ttl: Optional[int] = None) -> bool:
        """계산 결과 저장"""
        try:
            # 메모리 캐시에 저장
            self.calculation_cache.set(calc_id, data, ttl)

            # 디스크 캐시에 저장
            await self.disk_cache.set(f"calc_{calc_id}", data, ttl)

            return True
        except Exception as e:
            self.logger.error(f"Error caching calculation {calc_id}: {e}")
            return False

    async def get_prediction(self, structure_id: str) -> Optional[Dict]:
        """예측 결과 조회"""
        return self.prediction_cache.get(structure_id)

    async def set_prediction(self,
                             structure_id: str,
                             data: Dict,
                             ttl: Optional[int] = None) -> bool:
        """예측 결과 저장"""
        try:
            self.prediction_cache.set(structure_id, data, ttl)
            return True
        except Exception as e:
            self.logger.error(f"Error caching prediction for {structure_id}: {e}")
            return False

    async def clear_all(self):
        """모든 캐시 초기화"""
        # 메모리 캐시 초기화
        self.structure_cache.clear()
        self.calculation_cache.clear()
        self.prediction_cache.clear()

        # 디스크 캐시 초기화
        await self.disk_cache.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        memory_stats = {
            'structure_cache': self.structure_cache.get_stats(),
            'calculation_cache': self.calculation_cache.get_stats(),
            'prediction_cache': self.prediction_cache.get_stats()
        }

        disk_stats = await self.disk_cache.get_stats()

        return {
            'memory_cache': memory_stats,
            'disk_cache': disk_stats
        }

    async def cleanup(self):
        """캐시 정리"""
        # 메모리 캐시 정리
        self.structure_cache._cleanup_expired()
        self.calculation_cache._cleanup_expired()
        self.prediction_cache._cleanup_expired()

        # 디스크 캐시 정리
        await self.disk_cache._cleanup_cache()

    async def optimize(self):
        """캐시 최적화"""
        stats = await self.get_stats()

        # 메모리 사용량이 높은 경우 정리
        for cache_name, cache_stats in stats['memory_cache'].items():
            if cache_stats['total_entries'] > cache_stats['max_size'] * 0.9:
                await self.cleanup()
                break

        # 디스크 사용량이 높은 경우 정리
        if stats['disk_cache']['usage_percent'] > 90:
            await self.disk_cache._cleanup_cache()