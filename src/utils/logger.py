import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

class Logger:
    """
    A custom logger utility with timestamp, date, and file path support.
    Default log level is INFO.
    """

    def __init__(self, file_path: str, log_level: str = "INFO"):
        """
        Initialize the logger.

        Args:
            file_path (str): Path where log files will be stored
            log_level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is INFO.
        """
        self.file_path = file_path
        self.log_level = log_level.upper()

        log_dir = os.path.dirname(file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(f"strategy_logger_{os.path.basename(file_path)}")
        self.logger.setLevel(getattr(logging, self.log_level))

        self.logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(getattr(logging, self.log_level))
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level))
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _get_timestamp(self) -> str:
        """Get current timestamp in a readable format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.logger.critical(message)

    def log(self, level: str, message: str) -> None:
        """
        Log a message with specified level.

        Args:
            level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message (str): Message to log
        """
        level = level.upper()
        if hasattr(self.logger, level.lower()):
            getattr(self.logger, level.lower())(message)
        else:
            self.logger.warning(f"Invalid log level: {level}. Using INFO level.")
            self.logger.info(message)

    def get_log_file_path(self) -> str:
        """Get the current log file path."""
        return self.file_path

    def get_log_level(self) -> str:
        """Get the current log level."""
        return self.log_level

def create_logger(file_path: str, log_level: str = "INFO") -> Logger:
    """
    Create a logger instance.

    Args:
        file_path (str): Path where log files will be stored
        log_level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is INFO.

    Returns:
        Logger: Logger instance
    """
    return Logger(file_path, log_level)


if __name__ == "__main__":
    logger = create_logger("logs/strategy.log")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    debug_logger = create_logger("logs/debug.log", "DEBUG")
    debug_logger.debug("This debug message will be logged")

