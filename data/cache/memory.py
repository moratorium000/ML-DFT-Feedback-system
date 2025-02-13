from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
from collections import OrderedDict
import threading
import logging
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size: Optional[int] = None


class MemoryCache:
    """메모리 캐시 시스템"""

    def __init__(self,
                 max_size: int = 1000,
                 ttl: int = 3600,  # seconds
                 cleanup_interval: int = 300):  # seconds
        self.max_size = max_size
        self.ttl = ttl
        self.cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # 청소 스레드 시작
        self._start_cleanup_thread()

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            # TTL 검사
            if self._is_expired(entry):
                self._cache.pop(key)
                return None

            # 접근 정보 업데이트
            entry.last_accessed = datetime.now()
            entry.access_count += 1

            return entry.value

    def set(self,
            key: str,
            value: Any,
            ttl: Optional[int] = None) -> bool:
        """캐시에 값 저장"""
        with self._lock:
            # 캐시 크기 검사
            if len(self._cache) >= self.max_size:
                self._evict_entries()

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size=self._estimate_size(value)
            )

            self._cache[key] = entry
            return True

    def delete(self, key: str) -> bool:
        """캐시에서 값 삭제"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """캐시 전체 삭제"""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보"""
        with self._lock:
            total_entries = len(self._cache)
            total_size = sum(
                entry.size or 0 for entry in self._cache.values()
            )
            avg_access = np.mean([
                entry.access_count for entry in self._cache.values()
            ]) if total_entries > 0 else 0

            return {
                'total_entries': total_entries,
                'total_size': total_size,
                'average_access_count': avg_access,
                'hit_ratio': self._calculate_hit_ratio()
            }

    def _start_cleanup_thread(self):
        """청소 스레드 시작"""

        def cleanup_task():
            while True:
                self._cleanup_expired()
                threading.Event().wait(self.cleanup_interval)

        cleanup_thread = threading.Thread(
            target=cleanup_task,
            daemon=True
        )
        cleanup_thread.start()

    def _cleanup_expired(self):
        """만료된 엔트리 정리"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                del self._cache[key]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """엔트리 만료 여부 확인"""
        age = datetime.now() - entry.created_at
        return age.total_seconds() > self.ttl

    def _evict_entries(self):
        """캐시 엔트리 제거"""
        # LRU (Least Recently Used) 정책
        with self._lock:
            # 가장 오래 전에 접근한 항목부터 제거
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed
            )

            # 공간의 20% 확보
            entries_to_remove = len(self._cache) // 5
            for key, _ in sorted_entries[:entries_to_remove]:
                del self._cache[key]

    def _estimate_size(self, value: Any) -> int:
        """객체 크기 추정"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, tuple, dict)):
                return len(str(value))  # 근사값
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                return sys.getsizeof(value)
        except:
            return 0

    def _calculate_hit_ratio(self) -> float:
        """캐시 히트율 계산"""
        total_access = sum(
            entry.access_count for entry in self._cache.values()
        )
        if total_access == 0:
            return 0.0
        return len(self._cache) / total_access


class CacheManager:
    """캐시 관리자"""

    def __init__(self):
        self.structure_cache = MemoryCache(max_size=500)
        self.calculation_cache = MemoryCache(max_size=200)
        self.ml_prediction_cache = MemoryCache(max_size=300, ttl=1800)

    async def cache_structure(self,
                              structure_id: str,
                              structure_data: Dict) -> bool:
        """구조 데이터 캐싱"""
        return self.structure_cache.set(structure_id, structure_data)

    async def get_cached_structure(self,
                                   structure_id: str) -> Optional[Dict]:
        """캐시된 구조 데이터 조회"""
        return self.structure_cache.get(structure_id)

    async def cache_calculation(self,
                                calc_id: str,
                                calc_data: Dict) -> bool:
        """계산 결과 캐싱"""
        return self.calculation_cache.set(calc_id, calc_data)

    async def get_cached_calculation(self,
                                     calc_id: str) -> Optional[Dict]:
        """캐시된 계산 결과 조회"""
        return self.calculation_cache.get(calc_id)

    async def cache_prediction(self,
                               structure_id: str,
                               prediction: Dict) -> bool:
        """ML 예측 결과 캐싱"""
        return self.ml_prediction_cache.set(structure_id, prediction)

    async def get_cached_prediction(self,
                                    structure_id: str) -> Optional[Dict]:
        """캐시된 ML 예측 조회"""
        return self.ml_prediction_cache.get(structure_id)