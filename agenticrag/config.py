from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(ValueError):
    pass


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float") from exc


@dataclass
class Config:
    deepseek_api_key: str
    siliconflow_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_dims: int = 1536
    docs_dir: Path = Path("docs")
    chroma_dir: Path = Path(".chroma")
    source_cache_dir: Path = Path(".agenticrag_cache")
    max_calls: int = 15
    token_threshold: int = 128000
    token_warning_ratio: float = 0.9

    def validate(self) -> "Config":
        missing = []
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if not self.siliconflow_api_key:
            missing.append("SILICONFLOW_API_KEY")
        if missing:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")
        if self.embedding_dims <= 0:
            raise ConfigError("EMBEDDING_DIMS must be positive")
        if self.max_calls <= 0:
            raise ConfigError("MAX_CALLS must be positive")
        if self.token_threshold <= 0:
            raise ConfigError("TOKEN_THRESHOLD must be positive")
        if not 0 < self.token_warning_ratio < 1:
            raise ConfigError("TOKEN_WARNING_RATIO must be between 0 and 1")
        return self


def load_config(load_dotenv_file: bool = True) -> Config:
    if load_dotenv_file:
        load_dotenv()

    config = Config(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        siliconflow_base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        siliconflow_embedding_model=os.getenv(
            "SILICONFLOW_EMBEDDING_MODEL",
            "Qwen/Qwen3-Embedding-4B",
        ),
        embedding_dims=_int_env("EMBEDDING_DIMS", 1536),
        docs_dir=Path(os.getenv("DOCS_DIR", "docs")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", ".chroma")),
        source_cache_dir=Path(os.getenv("SOURCE_CACHE_DIR", ".agenticrag_cache")),
        max_calls=_int_env("MAX_CALLS", 15),
        token_threshold=_int_env("TOKEN_THRESHOLD", 128000),
        token_warning_ratio=_float_env("TOKEN_WARNING_RATIO", 0.9),
    )
    return config.validate()
