from pathlib import Path

import pytest

from agenticrag.config import Config, ConfigError, load_config


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


def test_config_validates_warning_ratio():
    with pytest.raises(ConfigError, match="TOKEN_WARNING_RATIO"):
        Config(
            deepseek_api_key="a",
            siliconflow_api_key="b",
            token_warning_ratio=1.5,
        ).validate()
