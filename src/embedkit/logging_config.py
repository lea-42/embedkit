import functools
import inspect
import logging
import logging.config
import time
from collections.abc import Callable
from typing import Any

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}


def setup_logging() -> None:
    logging.config.dictConfig(LOGGING_CONFIG)


def log_time(logger: logging.Logger) -> Callable:
    """Decorator that logs the duration of a function call at INFO level.

    Works with both sync and async functions.
    """
    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                result = await fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("%s completed in %.2fs", fn.__name__, elapsed)
                return result
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("%s completed in %.2fs", fn.__name__, elapsed)
                return result
            return sync_wrapper
    return decorator
