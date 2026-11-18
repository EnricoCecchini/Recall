import yaml
from pathlib import Path
from functools import lru_cache, cached_property
from typing import Literal
from pydantic import BaseModel, Field

import pathspec


ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = ROOT / "config.yaml"
IGNORE_LIST: Path = ROOT / ".ragignore"

class LLMConfig(BaseModel):
    provider: Literal["llama_cpp", "ollama", "openai", "anthropic"]
    model: str


class EmbeddingConfig(BaseModel):
    provider: Literal["llama_cpp", "sentence-transformers", "ollama", "openai"]
    model: str


class PathsConfig(BaseModel):
    source_data: str


class Config(BaseModel):
    paths: PathsConfig
    llm: LLMConfig
    embeddings: EmbeddingConfig
    ignore_patterns: list[str] = Field(default_factory=list)

    @cached_property
    def load_ignore_patterns(self) -> pathspec.GitIgnoreSpec:
        """Load ignore patterns into cache."""

        return pathspec.GitIgnoreSpec.from_lines(lines=self.lines)


def __read_ignore_patterns() -> pathspec.GitIgnoreSpec:
    """Read ignore patterns from .ragignore file."""

    lines = []
    if IGNORE_LIST.is_file():
        lines += IGNORE_LIST.read_text(encoding="utf-8").splitlines()

    return lines


@lru_cache
def get_config() -> Config:

    with CONFIG_FILE.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw["ignore_patterns"] = raw.get("ignore_patterns", []) + __read_ignore_patterns()

    cfg = Config.model_validate(raw)
    return cfg
