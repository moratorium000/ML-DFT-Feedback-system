import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import threading
from functools import wraps
import time
import traceback


class LoggerConfig:
    """로거 설정"""

    def __init__(self,
                 log_dir: Union[str, Path] = "logs",
                 level: int = logging.INFO,
                 max_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                 date_format: str = "%Y-%m-%d %H:%M:%S"):
        self.log_dir = Path(log_dir)
        self.level = level
        self.max_size = max_size
        self.backup_count = backup_count
        self.format = format
        self.date_format = date_format


class CustomLogger(logging.Logger):
    """사용자 정의 로거"""

    def __init__(self, name: str, config: LoggerConfig):
        super().__init__(name)
        self.config = config
        self._setup_logger()

    def _setup_logger(self):
        """로거 설정"""
        self.setLevel(self.config.level)

        # 포매터 생성
        formatter = logging.Formatter(
            self.config.format,
            datefmt=self.config.date_format
        )

        # 파일 핸들러 설정
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            self.config.log_dir / f"{self.name}.log",
            maxBytes=self.config.max_size,
            backupCount=self.config.backup_count
        )
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)

    def log_exception(self, exc_info):
        """예외 로깅"""
        self.error(
            "Exception occurred",
            exc_info=exc_info,
            extra={
                'traceback': traceback.format_exception(*exc_info)
            }
        )

    def log_dict(self, data: dict, level: int = logging.INFO):
        """딕셔너리 로깅"""
        self.log(level, json.dumps(data, indent=2))


def get_logger(name: str, config: Optional[LoggerConfig] = None) -> CustomLogger:
    """로거 생성"""
    if config is None:
        config = LoggerConfig()

    logger = CustomLogger(name, config)
    return logger


def log_execution_time(logger: Optional[CustomLogger] = None):
    """실행 시간 로깅 데코레이터"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            execution_time = end_time - start_time
            log_message = f"{func.__name__} executed in {execution_time:.2f} seconds"

            if logger:
                logger.info(log_message)
            else:
                logging.info(log_message)

            return result

        return wrapper

    return decorator


def log_async_execution_time(logger: Optional[CustomLogger] = None):
    """비동기 실행 시간 로깅 데코레이터"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()

            execution_time = end_time - start_time
            log_message = f"{func.__name__} executed in {execution_time:.2f} seconds"

            if logger:
                logger.info(log_message)
            else:
                logging.info(log_message)

            return result

        return wrapper

    return decorator


class ThreadSafeLogger:
    """스레드 안전 로거"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._loggers = {}

    def get_logger(self,
                   name: str,
                   config: Optional[LoggerConfig] = None) -> CustomLogger:
        """스레드 안전 로거 생성"""
        with self._lock:
            if name not in self._loggers:
                self._loggers[name] = get_logger(name, config)
            return self._loggers[name]


class LogContext:
    """로깅 컨텍스트 매니저"""

    def __init__(self,
                 logger: CustomLogger,
                 context_name: str,
                 level: int = logging.INFO):
        self.logger = logger
        self.context_name = context_name
        self.level = level
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(
            self.level,
            f"Entering context: {self.context_name}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        if exc_type:
            self.logger.error(
                f"Error in context {self.context_name}: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
        else:
            self.logger.log(
                self.level,
                f"Exiting context: {self.context_name} "
                f"(execution time: {execution_time:.2f}s)"
            )