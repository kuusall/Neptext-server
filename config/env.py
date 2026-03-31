import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List


def _parse_csv_env(value: str) -> List[str]:
	return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
	app_name: str
	app_version: str
	api_prefix: str
	log_level: str
	cors_origins: List[str]
	sentiment_model_name: str
	model_dicts_dir: str
	default_top_k: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
	cors_raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
	cors_origins = ["*"] if cors_raw.strip() == "*" else _parse_csv_env(cors_raw)

	return Settings(
		app_name=os.getenv("APP_NAME", "Nepali Text Processing API"),
		app_version=os.getenv("APP_VERSION", "1.0.0"),
		api_prefix=os.getenv("API_PREFIX", ""),
		log_level=os.getenv("LOG_LEVEL", "INFO"),
		cors_origins=cors_origins,
		sentiment_model_name=os.getenv(
			"SENTIMENT_MODEL_NAME", "cardiffnlp/twitter-xlm-roberta-base-sentiment"
		),
		model_dicts_dir=os.getenv("MODEL_DICTS_DIR", "model_dicts"),
		default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
	)
