"""Typed application settings.

Every tunable in the system is declared here and read from the environment (or
``.env``).  Nothing downstream is allowed to hard-code a threshold: the
roadmap's rule is that entry/exit levels, batch sizes and storage caps are
*configuration*, not constants buried in code.

Secrets are held as ``SecretStr`` so that an accidental ``print(settings)`` or a
logged traceback cannot leak the Atlas connection string.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SSE_",
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ database
    mongodb_uri: SecretStr = Field(
        default=SecretStr(""),
        description="mongodb+srv:// connection string. Password must be percent-encoded.",
    )
    db_name: str = "sse_market"
    server_selection_timeout_ms: int = 10_000

    # ------------------------------------------------------------------ ingestion
    history_start: dt.date = dt.date(2010, 1, 1)
    max_workers: int = Field(default=6, ge=1, le=32)
    requests_per_minute: int = Field(default=90, ge=1)
    request_timeout: int = Field(default=30, ge=5)
    max_retries: int = Field(default=4, ge=0, le=10)
    backoff_base: float = Field(default=1.5, ge=1.0)
    backoff_cap: float = Field(default=60.0, ge=1.0)
    #: days of history re-downloaded by the incremental updater so that Yahoo's
    #: silent corrections to recent bars are absorbed rather than frozen in.
    update_overlap_days: int = Field(default=10, ge=1)

    # ------------------------------------------------------------------ storage guard
    storage_cap_bytes: int = 536_870_912  # Atlas M0 = 512 MiB, indexes included
    storage_halt_fraction: float = Field(default=0.90, gt=0.0, le=1.0)
    storage_check_every: int = Field(default=50, ge=1, description="tickers between dbStats polls")

    # ------------------------------------------------------------------ universe
    #: Screener page size. Yahoo caps this at 250 and silently truncates deep pages,
    #: which is why `ingest.universe` bisects market-cap bands instead of paging blindly.
    screener_page_size: int = Field(default=250, ge=1, le=250)
    screener_max_offset: int = Field(default=1_000, ge=250)
    #: SSE code ranges used by the independent cross-check probe.
    #: 600/601/603/605 main board, 688/689 STAR, 900 B-shares (CNY-quoted, USD-traded).
    code_ranges: list[str] = ["600", "601", "603", "605", "688", "689", "900"]
    exchange_code: str = "SHH"  # Yahoo's exchange key for Shanghai
    yahoo_suffix: str = ".SS"

    # ------------------------------------------------------------------ api
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    max_range_days: int = Field(default=7_300, ge=1, description="~20y guard on one request")
    max_batch_tickers: int = Field(default=300, ge=1)
    api_title: str = "SSE Daily Market Data API"
    api_version: str = "1.0.0"

    # ------------------------------------------------------------------ research
    # Fixed in the pre-coding checklist so that no threshold is ever a magic
    # number inside a strategy function.  The sensitivity sweep in Phase 2
    # varies these; the defaults are what the headline result uses.
    formation_window: int = Field(default=252, ge=30, description="sessions used to fit a pair")
    trading_window: int = Field(default=63, ge=5, description="sessions traded before refitting")
    zscore_window: int = Field(default=63, ge=5, description="rolling window for the spread z")
    entry_z: float = Field(default=2.0, gt=0)
    exit_z: float = Field(default=0.5, ge=0)
    stop_z: float = Field(default=3.5, gt=0)
    max_holding_days: int = Field(default=30, ge=1)
    min_history_sessions: int = Field(default=756, ge=1, description="~3y to enter the universe")
    max_half_life: float = Field(default=30.0, gt=0, description="sessions; slower pairs dropped")
    min_half_life: float = Field(default=1.0, gt=0)
    fdr_alpha: float = Field(default=0.05, gt=0, lt=1, description="Benjamini-Hochberg level")
    # Frictions (China-specific): stamp duty is charged on sells only.  Verify the
    # current rate before quoting it in the deck — it was cut to 5 bp in Aug 2023.
    commission_bps: float = Field(default=2.5, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)
    stamp_duty_bps_sell: float = Field(default=5.0, ge=0)
    borrow_cost_bps_annual: float = Field(default=850.0, ge=0)

    # ------------------------------------------------------------------ logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    log_json: bool = False

    @field_validator("history_start", mode="before")
    @classmethod
    def _parse_date(cls, v: object) -> object:
        if isinstance(v, str):
            return dt.date.fromisoformat(v)
        return v

    @property
    def storage_halt_bytes(self) -> int:
        return int(self.storage_cap_bytes * self.storage_halt_fraction)

    def require_uri(self) -> str:
        """Return the connection string, failing loudly if it was never set."""
        uri = self.mongodb_uri.get_secret_value()
        if not uri or any(marker in uri for marker in ("<user>", "<url-encoded-password>", "<cluster>")):
            raise RuntimeError(
                "SSE_MONGODB_URI is missing or contains placeholder values. Set the real MongoDB "
                "Atlas connection string in .env; the password must be percent-encoded."
            )
        return uri


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton so the `.env` file is parsed exactly once."""
    return Settings()
