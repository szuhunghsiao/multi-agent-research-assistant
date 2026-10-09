"""Runtime settings loaded from environment / .env. Fails fast on missing keys."""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    researcher_api_key: SecretStr
    writer_api_key: SecretStr
    critic_api_key: SecretStr
    tavily_api_key: SecretStr

    model_name: str = "gemini-3.5-flash-lite"
    llm_timeout_s: float = 30.0
    max_iterations: int = 3   # Researcher tool-calling rounds (safety net)
    max_revisions: int = 3    # Writer<->Critic revise loop cap
    search_max_results: int = 3
    workers: int = 2          # concurrent graph runs (= concurrent LLM pressure)
    max_queue: int = 20       # pending jobs before POST returns 503
    retry_max_attempts: int = 4  # per LLM/search call, incl. the first try
    job_timeout_s: float = 300.0  # whole-job deadline across all calls and retries

@lru_cache
def get_settings() -> Settings:
    return Settings()