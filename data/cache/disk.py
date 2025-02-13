from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import pickle
import shutil
import hashlib
from datetime import datetime
import fcntl
import logging
from dataclasses import dataclass


@dataclass
class DiskCacheConfig:
    """디스크 캐시 설정"""
    base_dir: Path
    max_size_gb: float = 10.0  # 최대 캐시 크기 (GB)
    cleanup_threshold: float = 0.9  # 청소 시작 임계값 (90%)
    file_permissions: int = 0o644  # 파일 권한
    compression: bool = True  # 압축 사용 여부


class DiskCache:
    """디스크 캐시 시스템"""

    def __init__(self, config: DiskCacheConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 캐시 디렉토리 초기화
        self.cache_dir = self.config.base_dir / "cache"
        self.meta_dir = self.config.base_dir / "metadata"
        self._initialize_directories()

    def _initialize_directories(self):
        """디렉토리 초기화"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 로드"""
        file_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        if not file_path.exists() or not meta_path.exists():
            return None

        try:
            # 파일 락 획득
            with self._file_lock(file_path):
                # 메타데이터 검사
                metadata = self._load_metadata(key)
                if self._is_expired(metadata):
                    self._remove_cache_entry(key)
                    return None

                # 데이터 로드
                with open(file_path, 'rb') as f:
                    data = pickle.load(f) if not self.config.compression \
                        else self._decompress_data(f.read())

                # 접근 정보 업데이트
                self._update_access_info(key)

                return data

        except Exception as e:
            self.logger.error(f"Error loading cache for key {key}: {e}")
            return None

    def set(self,
            key: str,
            value: Any,
            ttl: Optional[int] = None) -> bool:
        """캐시에 데이터 저장"""
        try:
            # 공간 확보 필요 여부 확인
            if self._needs_cleanup():
                self._cleanup_cache()

            file_path = self._get_cache_path(key)
            meta_path = self._get_meta_path(key)

            # 데이터 저장
            with self._file_lock(file_path):
                data = pickle.dumps(value) if not self.config.compression \
                    else self._compress_data(value)

                with open(file_path, 'wb') as f:
                    f.write(data)

                # 메타데이터 저장
                metadata = {
                    'created_at': datetime.now().isoformat(),
                    'last_accessed': datetime.now().isoformat(),
                    'size': len(data),
                    'ttl': ttl,
                    'access_count': 0,
                    'checksum': self._calculate_checksum(data)
                }

                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)

            return True

        except Exception as e:
            self.logger.error(f"Error setting cache for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """캐시 항목 삭제"""
        try:
            self._remove_cache_entry(key)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting cache for key {key}: {e}")
            return False

    def clear(self):
        """캐시 전체 삭제"""
        try:
            shutil.rmtree(self.cache_dir)
            shutil.rmtree(self.meta_dir)
            self._initialize_directories()
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """캐시 파일 경로"""
        hashed_key = self._hash_key(key)
        return self.cache_dir / f"{hashed_key}.cache"

    def _get_meta_path(self, key: str) -> Path:
        """메타데이터 파일 경로"""
        hashed_key = self._hash_key(key)
        return self.meta_dir / f"{hashed_key}.meta"

    def _hash_key(self, key: str) -> str:
        """키 해싱"""
        return hashlib.sha256(key.encode()).hexdigest()

    def _calculate_checksum(self, data: bytes) -> str:
        """체크섬 계산"""
        return hashlib.md5(data).hexdigest()

    def _is_expired(self, metadata: Dict) -> bool:
        """만료 여부 확인"""
        if 'ttl' not in metadata or metadata['ttl'] is None:
            return False

        created_at = datetime.fromisoformat(metadata['created_at'])
        age = (datetime.now() - created_at).total_seconds()
        return age > metadata['ttl']

    def _needs_cleanup(self) -> bool:
        """청소 필요 여부 확인"""
        current_size = sum(f.stat().st_size for f in self.cache_dir.glob('*'))
        max_size = self.config.max_size_gb * 1024 * 1024 * 1024
        return current_size > (max_size * self.config.cleanup_threshold)

    def _cleanup_cache(self):
        """캐시 청소"""
        # LRU 정책으로 오래된 항목 제거
        entries = []
        for meta_file in self.meta_dir.glob('*.meta'):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                entries.append((
                    meta_file.stem,
                    datetime.fromisoformat(metadata['last_accessed'])
                ))
            except Exception:
                continue

        # 정렬 및 제거
        entries.sort(key=lambda x: x[1])
        for key, _ in entries[:len(entries) // 4]:  # 25% 제거
            self._remove_cache_entry(key)

    def _remove_cache_entry(self, key: str):
        """캐시 항목 제거"""
        file_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)

        if file_path.exists():
            file_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    @staticmethod
    def _file_lock(file_path: Path):
        """파일 락 컨텍스트 매니저"""

        class FileLock:
            def __init__(self, path):
                self.path = path
                self.fd = None

            def __enter__(self):
                self.fd = open(self.path, 'a+b')
                fcntl.flock(self.fd, fcntl.LOCK_EX)
                return self.fd

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.fd:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
                    self.fd.close()

        return FileLock(file_path)

    def _compress_data(self, data: Any) -> bytes:
        """데이터 압축"""
        import zlib
        return zlib.compress(pickle.dumps(data))

    def _decompress_data(self, data: bytes) -> Any:
        """데이터 압축 해제"""
        import zlib
        return pickle.loads(zlib.decompress(data))