"""
Indian Constitution AI - Application Configuration

All settings come from environment variables (or ``backend/.env``). Paths are resolved
relative to the backend directory, not the process working directory, so the app and
the scripts behave the same wherever they are launched from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=str(BACKEND_DIR / ".env"), env_file_encoding="utf-8",
                                      case_sensitive=False, extra="ignore", populate_by_name=True)

    # --- Application ---
    app_name: str = Field(default="Indian Constitution AI")
    app_env: str = Field(default="development")
    app_debug: bool = Field(default=False)
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    app_secret_key: str = Field(default="change-this-to-a-random-secret-key")
    admin_api_token: Optional[str] = Field(default=None)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")          # json | text
    store_query_text: bool = Field(default=False)     # keep raw questions in debug traces

    # --- Database ---
    database_url: str = Field(default="sqlite:///./data/app.db")
    data_dir_setting: str = Field(default="./data", validation_alias=AliasChoices("DATA_DIR", "data_dir_setting"))
    backup_dir_setting: Optional[str] = Field(default=None, validation_alias=AliasChoices("BACKUP_DIR", "backup_dir_setting"))

    # --- LLM Provider (Gemini) ---
    llm_provider: str = Field(default="gemini")
    gemini_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "LLM_API_KEY", "gemini_api_key"))
    gemini_model: str = Field(default="gemini-3.8-flash", validation_alias=AliasChoices("GEMINI_MODEL", "LLM_MODEL", "gemini_model"))
    gemini_fallback_models: str = Field(default="gemini-3.5-flash-lite")
    gemini_classifier_model: Optional[str] = Field(default=None)
    gemini_embedding_model: str = Field(default="gemini-embedding-001")
    gemini_embedding_dimensions: int = Field(default=768, validation_alias=AliasChoices("GEMINI_EMBEDDING_DIMENSIONS", "EMBEDDING_DIMENSIONS", "gemini_embedding_dimensions"))
    gemini_timeout: float = Field(default=45.0, validation_alias=AliasChoices("GEMINI_TIMEOUT", "LLM_TIMEOUT", "gemini_timeout"))
    gemini_max_retries: int = Field(default=2)
    gemini_temperature: float = Field(default=0.2, validation_alias=AliasChoices("GEMINI_TEMPERATURE", "LLM_TEMPERATURE", "gemini_temperature"))
    gemini_max_output_tokens: int = Field(default=3072, validation_alias=AliasChoices("GEMINI_MAX_OUTPUT_TOKENS", "LLM_MAX_TOKENS", "gemini_max_output_tokens"))
    gemini_thinking_level: str = Field(default="low")

    # --- Embeddings / vector retrieval ---
    enable_embeddings: bool = Field(default=True)
    embedding_batch_size: int = Field(default=50)
    embedding_texts_per_minute: int = Field(default=90)   # free-tier pacing; 0 = unlimited

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=12)
    retrieval_weight_exact: float = Field(default=3.0)
    retrieval_weight_graph: float = Field(default=1.6)
    retrieval_weight_bm25: float = Field(default=1.0)
    retrieval_weight_vector: float = Field(default=1.0)
    retrieval_rrf_k: int = Field(default=60)
    query_classifier: str = Field(default="hybrid")   # rules | hybrid (rules + Gemini, validated)

    # --- Knowledge Sources ---
    default_knowledge_source: str = Field(default="constitution_of_india")
    enable_default_knowledge: bool = Field(default=True)
    enable_user_documents: bool = Field(default=True)
    enable_knowledge_graph: bool = Field(default=True)
    enable_multimodal: bool = Field(default=True)

    # --- Knowledge Graph (Neo4j) ---
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: Optional[str] = Field(default=None)
    neo4j_database: str = Field(default="neo4j")
    enable_neo4j: bool = Field(default=False)

    # --- File Upload ---
    max_upload_size_mb: int = Field(default=50)
    upload_dir: str = Field(default="./data/uploads")
    allowed_extensions: str = Field(default="pdf,png,jpg,jpeg,tiff,bmp")

    # --- OCR ---
    tesseract_cmd: str = Field(default="tesseract")
    enable_ocr: bool = Field(default=True)

    # --- CORS ---
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:8000")

    # --- Rate Limiting ---
    rate_limit_requests: int = Field(default=60)
    rate_limit_window: int = Field(default=60)

    @field_validator("llm_provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        value = (value or "gemini").strip().lower()
        return "gemini" if value in {"google", "gemini"} else value

    # ------------------------------------------------------------------ paths
    @staticmethod
    def _resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (BACKEND_DIR / candidate).resolve()

    @property
    def data_dir(self) -> Path:
        return self._resolve(self.data_dir_setting)

    @property
    def constitution_dir(self) -> Path:
        return self.data_dir / "constitution"

    @property
    def statutes_dir(self) -> Path:
        return self.data_dir / "statutes"

    @property
    def backup_dir(self) -> Path:
        return self._resolve(self.backup_dir_setting) if self.backup_dir_setting else self.data_dir / "backups"

    @property
    def upload_path(self) -> Path:
        return self._resolve(self.upload_dir)

    @property
    def resolved_database_url(self) -> str:
        url = self.database_url
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////") and ":memory:" not in url:
            relative = url[len("sqlite:///"):]
            if not Path(relative).is_absolute():
                return "sqlite:///" + str(self._resolve(relative)).replace("\\", "/")
        return url

    # ------------------------------------------------------------------ derived
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extension_list(self) -> list[str]:
        return [e.strip().lower() for e in self.allowed_extensions.split(",")]

    @property
    def fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.gemini_fallback_models.split(",") if m.strip()]

    @property
    def is_llm_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def is_neo4j_configured(self) -> bool:
        return self.enable_neo4j and bool(self.neo4j_password)

    def validation_problems(self) -> list[str]:
        """Configuration problems reported at startup and by the readiness endpoint."""
        problems = []
        if self.llm_provider != "gemini":
            problems.append(f"LLM_PROVIDER={self.llm_provider!r} is not supported; use 'gemini'.")
        if not self.gemini_api_key:
            problems.append("GEMINI_API_KEY is not set; answers will use the evidence-only fallback.")
        if self.app_env == "production" and self.app_secret_key.startswith("change-this"):
            problems.append("APP_SECRET_KEY must be changed in production.")
        if self.app_env == "production" and not self.admin_api_token:
            problems.append("ADMIN_API_TOKEN is not set; admin endpoints are disabled in production.")
        if self.enable_neo4j and not self.neo4j_password:
            problems.append("ENABLE_NEO4J=true but NEO4J_PASSWORD is not set.")
        if self.gemini_embedding_dimensions not in {128, 256, 512, 768, 1536, 3072}:
            problems.append("GEMINI_EMBEDDING_DIMENSIONS should be one of 128-3072 supported by the model.")
        return problems


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    global _settings
    _settings = None
