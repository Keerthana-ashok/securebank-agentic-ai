"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SECUREBANK_", extra="ignore")

    # Accept the standard Anthropic variable name without the SECUREBANK_ prefix.
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "SECUREBANK_ANTHROPIC_API_KEY"),
    )

    model: str = "claude-opus-5-5"
    # Claude Opus 5.5 defaults to "medium"; set explicitly so behavior is predictable.
    effort: str = "medium"
    max_tokens: int = 16000

    audit_log_path: str = "audit.log"


@lru_cache
def get_settings() -> Settings:
    return Settings()
