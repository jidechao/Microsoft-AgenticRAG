from pathlib import Path

import pytest

from agenticrag.config import Config, ConfigError, load_config


CONFIG_ENV_VARS = [
    "DEEPSEEK_API_KEY",
    "SILICONFLOW_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "SILICONFLOW_BASE_URL",
    "SILICONFLOW_EMBEDDING_MODEL",
    "EMBEDDING_DIMS",
    "DOCS_DIR",
    "CHROMA_DIR",
    "SOURCE_CACHE_DIR",
    "MAX_CALLS",
    "TOKEN_THRESHOLD",
    "TOKEN_WARNING_RATIO",
]


@pytest.fixture(autouse=True)
def isolate_config_env(monkeypatch):
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_load_config_uses_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "silicon-key")

    config = load_config(load_dotenv_file=False)

    assert config.deepseek_api_key == "deepseek-key"
    assert config.siliconflow_api_key == "silicon-key"
    assert config.deepseek_base_url == "https://api.deepseek.com"
    assert config.deepseek_model == "deepseek-chat"
    assert config.siliconflow_embedding_model == "Qwen/Qwen3-Embedding-4B"
    assert config.embedding_dims == 1536
    assert config.docs_dir == Path("docs")
    assert config.chroma_dir == Path(".chroma")
    assert config.source_cache_dir == Path(".agenticrag_cache")
    assert config.max_calls == 15
    assert config.token_threshold == 128000
    assert config.token_warning_ratio == 0.9


def test_load_config_requires_api_keys(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    with pytest.raises(ConfigError) as exc:
        load_config(load_dotenv_file=False)

    assert "DEEPSEEK_API_KEY" in str(exc.value)
    assert "SILICONFLOW_API_KEY" in str(exc.value)


def test_load_config_uses_defaults_for_empty_optional_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "silicon-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "")
    monkeypatch.setenv("DEEPSEEK_MODEL", "   ")
    monkeypatch.setenv("SILICONFLOW_BASE_URL", "\t")
    monkeypatch.setenv("SILICONFLOW_EMBEDDING_MODEL", "\n")
    monkeypatch.setenv("DOCS_DIR", "")
    monkeypatch.setenv("CHROMA_DIR", "   ")
    monkeypatch.setenv("SOURCE_CACHE_DIR", "\t")

    config = load_config(load_dotenv_file=False)

    assert config.deepseek_base_url == "https://api.deepseek.com"
    assert config.deepseek_model == "deepseek-chat"
    assert config.siliconflow_base_url == "https://api.siliconflow.cn/v1"
    assert config.siliconflow_embedding_model == "Qwen/Qwen3-Embedding-4B"
    assert config.docs_dir == Path("docs")
    assert config.chroma_dir == Path(".chroma")
    assert config.source_cache_dir == Path(".agenticrag_cache")


def test_load_config_rejects_whitespace_api_keys(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "   ")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "\t")

    with pytest.raises(ConfigError) as exc:
        load_config(load_dotenv_file=False)

    assert "DEEPSEEK_API_KEY" in str(exc.value)
    assert "SILICONFLOW_API_KEY" in str(exc.value)


def test_config_validates_warning_ratio():
    with pytest.raises(ConfigError, match="TOKEN_WARNING_RATIO"):
        Config(
            deepseek_api_key="a",
            siliconflow_api_key="b",
            token_warning_ratio=1.5,
        ).validate()
