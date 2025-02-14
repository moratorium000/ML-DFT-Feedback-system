from typing import Dict, List, Optional, Union, Any, BinaryIO
from pathlib import Path
import json
import pickle
import shutil
import hashlib
from datetime import datetime
import fcntl
import logging
import zlib
import os
import asyncio
from dataclasses import dataclass


@dataclass
class DiskCacheConfig:
    """디스크 캐시 설정"""
    base_path: Path
    max_size_gb: float = 10.0
    cleanup_threshold: float = 0.9
    file_permissions: int = 0o644
    compression: bool = True
    chunk_size: int = 8192
    cleanup_interval: int = 3600


class DiskCache:
    """디스크 기반 캐시 시스템"""

    def __init__(self, config: DiskCacheConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 캐시 디렉토리 초기화
        self.cache_dir = self.config.base_path / "data"
        self.meta_dir = self.config.base_path / "metadata"
        self._initialize_directories()

        # 청소 작업 스케줄링
        self._schedule_cleanup()

    def _initialize_directories(self):
        """디렉토리 구조 초기화"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        if not cache_path.exists() or not meta_path.exists():
            return None

        try:
            metadata = await self._load_metadata(key)
            if self._is_expired(metadata):
                await self._remove_cache_entry(key)
                return None

            async with self._file_lock(cache_path):
                data = await self._read_cache_file(cache_path)
                await self._update_access_info(key)
                return data

        except Exception as e:
            self.logger.error(f"Error retrieving cache for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시에 데이터 저장"""
        try:
            if await self._needs_cleanup():
                await self._cleanup_cache()

            cache_path = self._get_cache_path(key)
            meta_path = self._get_meta_path(key)

            async with self._file_lock(cache_path):
                # 데이터 저장
                compressed_data = self._compress_data(value)
                await self._write_cache_file(cache_path, compressed_data)

                # 메타데이터 저장
                metadata = {
                    'key': key,
                    'created_at': datetime.now().isoformat(),
                    'last_accessed': datetime.now().isoformat(),
                    'ttl': ttl,
                    'size': len(compressed_data),
                    'checksum': hashlib.md5(compressed_data).hexdigest()
                }
                await self._save_metadata(key, metadata)

            return True

        except Exception as e:
            self.logger.error(f"Error setting cache for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """캐시 항목 삭제"""
        try:
            await self._remove_cache_entry(key)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting cache entry {key}: {e}")
            return False

    async def clear(self):
        """캐시 전체 삭제"""
        try:
            shutil.rmtree(self.cache_dir)
            shutil.rmtree(self.meta_dir)
            self._initialize_directories()
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 조회"""
        total_size = await self._get_cache_size()
        num_entries = len(list(self.cache_dir.glob('*')))

        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'num_entries': num_entries,
            'usage_percent': (total_size / (self.config.max_size_gb * 1024 * 1024 * 1024)) * 100
        }

    def _compress_data(self, data: Any) -> bytes:
        """데이터 압축"""
        if self.config.compression:
            pickled_data = pickle.dumps(data)
            return zlib.compress(pickled_data)
        return pickle.dumps(data)

    def _decompress_data(self, data: bytes) -> Any:
        """데이터 압축 해제"""
        if self.config.compression:
            decompressed_data = zlib.decompress(data)
            return pickle.loads(decompressed_data)
        return pickle.loads(data)

    def _get_cache_path(self, key: str) -> Path:
        """캐시 파일 경로 생성"""
        hashed_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.cache"

    def _get_meta_path(self, key: str) -> Path:
        """메타데이터 파일 경로 생성"""
        hashed_key = hashlib.sha256(key.encode()).hexdigest()
        return self.meta_dir / f"{hashed_key}.meta"

    async def _load_metadata(self, key: str) -> Dict:
        """메타데이터 로드"""
        meta_path = self._get_meta_path(key)
        async with aiofiles.open(meta_path, 'r') as f:
            content = await f.read()
            return json.loads(content)

    async def _save_metadata(self, key: str, metadata: Dict):
        """메타데이터 저장"""
        meta_path = self._get_meta_path(key)
        async with aiofiles.open(meta_path, 'w') as f:
            await f.write(json.dumps(metadata))

    def _is_expired(self, metadata: Dict) -> bool:
        """만료 여부 확인"""
        if 'ttl' not in metadata or metadata['ttl'] is None:
            return False

        created_at = datetime.fromisoformat(metadata['created_at'])
        age = (datetime.now() - created_at).total_seconds()
        return age > metadata['ttl']

    async def _get_cache_size(self) -> int:
        """전체 캐시 크기 계산"""
        total_size = 0
        for file_path in self.cache_dir.glob('*'):
            total_size += file_path.stat().st_size
        return total_size

    async def _needs_cleanup(self) -> bool:
        """청소 필요 여부 확인"""
        current_size = await self._get_cache_size()
        max_size = self.config.max_size_gb * 1024 * 1024 * 1024
        return current_size > (max_size * self.config.cleanup_threshold)

    async def _cleanup_cache(self):
        """오래된 캐시 정리"""
        entries = []
        for meta_file in self.meta_dir.glob('*.meta'):
            metadata = await self._load_metadata(meta_file.stem)
            entries.append((
                meta_file.stem,
                datetime.fromisoformat(metadata['last_accessed'])
            ))

        # 가장 오래된 항목부터 정리
        entries.sort(key=lambda x: x[1])
        for key, _ in entries[:len(entries) // 4]:  # 25% 정리
            await self._remove_cache_entry(key)

    async def _schedule_cleanup(self):
        """주기적 청소 작업 스케줄링"""
        while True:
            await asyncio.sleep(self.config.cleanup_interval)
            if await self._needs_cleanup():
                await self._cleanup_cache()

    @staticmethod
    async def _file_lock(file_path: Path):
        """파일 락 컨텍스트 매니저"""

        class FileLock:
            def __init__(self, path):
                self.path = path
                self.fd = None

            async def __aenter__(self):
                self.fd = await aiofiles.open(self.path, 'a+b')
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)
                return self.fd

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.fd:
                    fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                    await self.fd.close()

        return FileLock(file_path)

    async def _remove_cache_entry(self, key: str):
        """캐시 항목 제거"""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        if cache_path.exists():
            os.unlink(cache_path)
        if meta_path.exists():
            os.unlink(meta_path)

    async def _read_cache_file(self, cache_path: Path) -> Any:
        """캐시 파일 읽기"""
        async with aiofiles.open(cache_path, 'rb') as f:
            data = await f.read()
            return self._decompress_data(data)

    async def _write_cache_file(self, cache_path: Path, data: bytes):
        """캐시 파일 쓰기"""
        async with aiofiles.open(cache_path, 'wb') as f:
            await f.write(data)