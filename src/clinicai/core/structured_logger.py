"""
Structured logging utilities for comprehensive application logging
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredLogger:
    """
    Structured logger that outputs JSON logs for easy parsing and querying
    """

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Add JSON formatter if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)

    def log(self, level: str, message: str, **kwargs):
        """Log with structured data"""
        log_data = {"message": message, **kwargs}

        if level == "info":
            self.logger.info(json.dumps(log_data))
        elif level == "warning":
            self.logger.warning(json.dumps(log_data))
        elif level == "error":
            self.logger.error(json.dumps(log_data))
        elif level == "debug":
            self.logger.debug(json.dumps(log_data))
        elif level == "critical":
            self.logger.critical(json.dumps(log_data))

    def info(self, message: str, **kwargs):
        """Log info level"""
        self.log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning level"""
        self.log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error level"""
        self.log("error", message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug level"""
        self.log("debug", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical level"""
        self.log("critical", message, **kwargs)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger"""
    return StructuredLogger(name)
