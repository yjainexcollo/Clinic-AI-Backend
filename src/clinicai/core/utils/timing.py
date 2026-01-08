"""
Performance timing utilities for transcription pipeline.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger("clinicai")
# Ensure timing logs are visible at INFO level even if the parent logger is stricter
if logger.level > logging.INFO or logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
logger.propagate = True


class TimingContext:
    """Context manager for timing operations with detailed metrics."""

    def __init__(self, stage_name: str, logger_instance: Optional[logging.Logger] = None):
        self.stage_name = stage_name
        self.logger = logger_instance or logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.input_size: Optional[int] = None
        self.output_size: Optional[int] = None
        self.metadata: Dict[str, Any] = {}

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"[TIMING START] {self.stage_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        # Build log message
        log_parts = [
            f"[TIMING END] {self.stage_name}",
            f"Duration: {self.duration:.3f}s",
        ]

        if self.input_size is not None:
            log_parts.append(f"Input: {self._format_size(self.input_size)}")

        if self.output_size is not None:
            log_parts.append(f"Output: {self._format_size(self.output_size)}")

        if self.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            log_parts.append(f"Metadata: {meta_str}")

        self.logger.info(" | ".join(log_parts))

        return False  # Don't suppress exceptions

    def set_input_size(self, size: int):
        """Set input size for logging."""
        self.input_size = size

    def set_output_size(self, size: int):
        """Set output size for logging."""
        self.output_size = size

    def add_metadata(self, **kwargs):
        """Add metadata to timing log."""
        self.metadata.update(kwargs)

    @staticmethod
    def _format_size(size: int) -> str:
        """Format size in human-readable format."""
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.2f}KB"
        else:
            return f"{size / (1024 * 1024):.2f}MB"


@contextmanager
def timing(stage_name: str, logger_instance: Optional[logging.Logger] = None):
    """Simple timing context manager."""
    with TimingContext(stage_name, logger_instance) as ctx:
        yield ctx
