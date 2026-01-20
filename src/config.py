"""Configuration management for Polymarket Wallet Analyzer."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """API configuration settings."""

    base_url: str = field(
        default_factory=lambda: os.getenv(
            "POLYMARKET_API_URL",
            "https://gamma-api.polymarket.com"
        )
    )
    clob_url: str = field(
        default_factory=lambda: os.getenv(
            "POLYMARKET_CLOB_API_URL",
            "https://clob.polymarket.com"
        )
    )
    rate_limit_calls: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_CALLS", "10"))
    )
    rate_limit_period: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PERIOD", "60"))
    )
    timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.5


@dataclass
class CacheConfig:
    """Cache configuration settings."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true"
    )
    ttl: int = field(
        default_factory=lambda: int(os.getenv("CACHE_TTL", "3600"))
    )
    directory: Path = field(default_factory=lambda: Path(".cache"))


@dataclass
class LogConfig:
    """Logging configuration settings."""

    level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    file: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("LOG_FILE", "outputs/analyzer.log"))
    )
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class ExportConfig:
    """Export configuration settings."""

    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DEFAULT_OUTPUT_DIR", "outputs"))
    )
    default_format: str = field(
        default_factory=lambda: os.getenv("DEFAULT_EXPORT_FORMAT", "json")
    )


@dataclass
class Config:
    """Main configuration container."""

    api: APIConfig = field(default_factory=APIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    log: LogConfig = field(default_factory=LogConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def __post_init__(self) -> None:
        """Initialize logging after config is loaded."""
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging based on settings."""
        log_level = getattr(logging, self.log.level.upper(), logging.INFO)

        handlers: list[logging.Handler] = [logging.StreamHandler()]

        if self.log.file:
            self.log.file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self.log.file))

        logging.basicConfig(
            level=log_level,
            format=self.log.format,
            handlers=handlers,
            force=True
        )


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        Config: The global configuration object.
    """
    return config


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: The logger name (typically __name__).

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
