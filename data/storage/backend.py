from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from enum import Enum
import asyncio
import aiohttp
import json
import logging
from dataclasses import dataclass


class StorageBackend(Enum):
    """저장소 백엔드 유형"""
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"
    FTP = "ftp"


@dataclass
class BackendConfig:
    """백엔드 설정"""
    backend_type: StorageBackend
    base_path: str
    credentials: Optional[Dict] = None
    max_connections: int = 10
    timeout: int = 30
    retry_attempts: int = 3
    chunk_size: int = 8192


class StorageBackendManager:
    """저장소 백엔드 관리자"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._backend = self._initialize_backend()

    def _initialize_backend(self) -> 'BaseBackend':
        """백엔드 초기화"""
        if self.config.backend_type == StorageBackend.LOCAL:
            return LocalBackend(self.config)
        elif self.config.backend_type == StorageBackend.S3:
            return S3Backend(self.config)
        elif self.config.backend_type == StorageBackend.AZURE:
            return AzureBackend(self.config)
        elif self.config.backend_type == StorageBackend.GCS:
            return GCSBackend(self.config)
        elif self.config.backend_type == StorageBackend.FTP:
            return FTPBackend(self.config)
        else:
            raise ValueError(f"Unsupported backend type: {self.config.backend_type}")

    async def store_file(self,
                         file_path: Union[str, Path],
                         data: Union[bytes, BinaryIO],
                         metadata: Optional[Dict] = None) -> bool:
        """파일 저장"""
        return await self._backend.store_file(file_path, data, metadata)

    async def retrieve_file(self,
                            file_path: Union[str, Path]) -> Optional[bytes]:
        """파일 검색"""
        return await self._backend.retrieve_file(file_path)

    async def delete_file(self,
                          file_path: Union[str, Path]) -> bool:
        """파일 삭제"""
        return await self._backend.delete_file(file_path)

    async def list_files(self,
                         prefix: Optional[str] = None) -> List[str]:
        """파일 목록 조회"""
        return await self._backend.list_files(prefix)


class BaseBackend:
    """기본 백엔드 클래스"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def store_file(self,
                         file_path: Union[str, Path],
                         data: Union[bytes, BinaryIO],
                         metadata: Optional[Dict] = None) -> bool:
        """파일 저장 구현"""
        raise NotImplementedError

    async def retrieve_file(self,
                            file_path: Union[str, Path]) -> Optional[bytes]:
        """파일 검색 구현"""
        raise NotImplementedError

    async def delete_file(self,
                          file_path: Union[str, Path]) -> bool:
        """파일 삭제 구현"""
        raise NotImplementedError

    async def list_files(self,
                         prefix: Optional[str] = None) -> List[str]:
        """파일 목록 조회 구현"""
        raise NotImplementedError


class LocalBackend(BaseBackend):
    """로컬 파일시스템 백엔드"""

    async def store_file(self,
                         file_path: Union[str, Path],
                         data: Union[bytes, BinaryIO],
                         metadata: Optional[Dict] = None) -> bool:
        try:
            path = Path(self.config.base_path) / file_path
            path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(data, bytes):
                async with aiofiles.open(path, 'wb') as f:
                    await f.write(data)
            else:
                async with aiofiles.open(path, 'wb') as f:
                    while chunk := await data.read(self.config.chunk_size):
                        await f.write(chunk)

            if metadata:
                meta_path = path.with_suffix(path.suffix + '.meta')
                async with aiofiles.open(meta_path, 'w') as f:
                    await f.write(json.dumps(metadata))

            return True

        except Exception as e:
            self.logger.error(f"Error storing file {file_path}: {e}")
            return False


class S3Backend(BaseBackend):
    """AWS S3 백엔드"""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        import boto3
        self.s3 = boto3.client('s3',
                               aws_access_key_id=config.credentials['access_key'],
                               aws_secret_access_key=config.credentials['secret_key'])
        self.bucket = config.base_path

    async def store_file(self,
                         file_path: Union[str, Path],
                         data: Union[bytes, BinaryIO],
                         metadata: Optional[Dict] = None) -> bool:
        try:
            extra_args = {'Metadata': metadata} if metadata else {}
            await self.s3.upload_fileobj(data, self.bucket, str(file_path),
                                         ExtraArgs=extra_args)
            return True
        except Exception as e:
            self.logger.error(f"Error storing file in S3 {file_path}: {e}")
            return False


class AzureBackend(BaseBackend):
    """Azure Blob Storage 백엔드"""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        from azure.storage.blob import BlobServiceClient
        self.blob_service = BlobServiceClient.from_connection_string(
            config.credentials['connection_string']
        )
        self.container = self.blob_service.get_container_client(
            config.base_path
        )


class GCSBackend(BaseBackend):
    """Google Cloud Storage 백엔드"""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        from google.cloud import storage
        self.client = storage.Client.from_service_account_json(
            config.credentials['service_account_path']
        )
        self.bucket = self.client.bucket(config.base_path)


class FTPBackend(BaseBackend):
    """FTP 백엔드"""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        import aioftp
        self.ftp_config = {
            'host': config.credentials['host'],
            'user': config.credentials['user'],
            'password': config.credentials['password']
        }