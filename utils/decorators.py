from typing import Any, Callable, Dict, Optional, Type
from functools import wraps
import time
import asyncio
import logging
from datetime import datetime
import traceback


def retry(max_attempts: int = 3,
          delay: float = 1.0,
          backoff: float = 2.0,
          exceptions: tuple = (Exception,)):
    """재시도 데코레이터"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def cache(ttl: Optional[int] = None):
    """캐시 데코레이터"""

    def decorator(func):
        cache_data = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str((args, tuple(sorted(kwargs.items()))))
            now = time.time()

            if key in cache_data:
                result, timestamp = cache_data[key]
                if ttl is None or now - timestamp < ttl:
                    return result

            result = func(*args, **kwargs)
            cache_data[key] = (result, now)
            return result

        return wrapper

    return decorator


def validate_input(**validators):
    """입력 검증 데코레이터"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 위치 인자를 키워드 인자로 변환
            func_args = func.__code__.co_varnames[:func.__code__.co_argcount]
            all_args = dict(zip(func_args, args))
            all_args.update(kwargs)

            # 검증 수행
            for param, validator in validators.items():
                if param in all_args:
                    value = all_args[param]
                    if not validator(value):
                        raise ValueError(
                            f"Invalid value for parameter {param}: {value}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_execution(logger: Optional[logging.Logger] = None,
                  level: int = logging.INFO):
    """실행 로깅 데코레이터"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            log = logger or logging.getLogger(func.__module__)

            try:
                log.log(level, f"Starting {func.__name__}")
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                log.log(level,
                        f"Completed {func.__name__} in {execution_time:.2f}s"
                        )
                return result
            except Exception as e:
                log.error(f"Error in {func.__name__}: {str(e)}",
                          exc_info=True)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            log = logger or logging.getLogger(func.__module__)

            try:
                log.log(level, f"Starting {func.__name__}")
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                log.log(level,
                        f"Completed {func.__name__} in {execution_time:.2f}s"
                        )
                return result
            except Exception as e:
                log.error(f"Error in {func.__name__}: {str(e)}",
                          exc_info=True)
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def deprecated(reason: str):
    """폐기 예정 표시 데코레이터"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.warning(
                f"{func.__name__} is deprecated: {reason}"
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def asynccontextmanager(func):
    """비동기 컨텍스트 매니저 데코레이터"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return AsyncContextManager(func, args, kwargs)

    return wrapper


class AsyncContextManager:
    """비동기 컨텍스트 매니저 구현"""

    def __init__(self, func, args, kwargs):
        self.gen = func(*args, **kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        try:
            return await self.gen.__anext__()
        except StopAsyncIteration:
            raise RuntimeError("Generator didn't yield")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.gen.__anext__()
        except StopAsyncIteration:
            return
        except Exception as e:
            if exc_type is None:
                raise RuntimeError(
                    "Generator didn't stop"
                ) from e
            else:
                raise


def singleton(cls):
    """싱글톤 패턴 데코레이터"""
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance