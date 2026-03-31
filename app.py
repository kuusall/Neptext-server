from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.env import get_settings
from schemas.api import (
	ErrorResponse,
	HealthResponse,
	SentimentRequest,
	SentimentResponse,
	SpellCorrectRequest,
	SpellCorrectResponse,
	WordPredictRequest,
	WordPredictResponse,
)
from services.gateway_service import APIGatewayService
from services.logger_service import get_api_logger, log_request


settings = get_settings()
logger = get_api_logger(settings.log_level)
gateway = APIGatewayService()

app = FastAPI(
	title=settings.app_name,
	version=settings.app_version,
	description=(
		"REST API for Nepali sentiment analysis, spell correction, and next-word prediction."
	),
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=settings.cors_origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


sentiment_router = APIRouter(
	prefix="/sentiment",
	tags=["Sentiment"],
	responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)

spell_router = APIRouter(
	prefix="/spell-correct",
	tags=["Spell Correction"],
	responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)

word_predict_router = APIRouter(
	prefix="/word-predict",
	tags=["Word Prediction"],
	responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)


@app.middleware("http")
async def request_logging_middleware(request, call_next):
	request_id = uuid4().hex
	started = perf_counter()

	try:
		response = await call_next(request)
		status_code = response.status_code
	except Exception:
		duration_ms = (perf_counter() - started) * 1000
		log_request(
			logger,
			request_id=request_id,
			method=request.method,
			path=request.url.path,
			status_code=500,
			duration_ms=duration_ms,
			client_ip=request.client.host if request.client else "unknown",
		)
		raise

	duration_ms = (perf_counter() - started) * 1000
	response.headers["X-Request-ID"] = request_id
	log_request(
		logger,
		request_id=request_id,
		method=request.method,
		path=request.url.path,
		status_code=status_code,
		duration_ms=duration_ms,
		client_ip=request.client.host if request.client else "unknown",
	)
	return response


@app.get("/", response_model=HealthResponse, tags=["Health"])
def root() -> HealthResponse:
	return HealthResponse(status="ok", message="Nepali text processing API is running")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
	return HealthResponse(status="ok", message="healthy")


@sentiment_router.post("", response_model=SentimentResponse)
def sentiment_endpoint(payload: SentimentRequest) -> SentimentResponse:
	try:
		result = gateway.analyze_sentiment(payload.text)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail="sentiment inference failed") from exc

	return SentimentResponse(**result)


@spell_router.post("", response_model=SpellCorrectResponse)
def spell_correct_endpoint(payload: SpellCorrectRequest) -> SpellCorrectResponse:
	try:
		result = gateway.correct_spelling(payload.text, payload.suggest_only)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except FileNotFoundError as exc:
		raise HTTPException(status_code=500, detail=f"model file missing: {exc}") from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail="spell correction failed") from exc

	return SpellCorrectResponse(**result)


@word_predict_router.post("", response_model=WordPredictResponse)
def word_predict_endpoint(payload: WordPredictRequest) -> WordPredictResponse:
	try:
		result = gateway.predict_next_word(payload.text, payload.top_k)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except FileNotFoundError as exc:
		raise HTTPException(status_code=500, detail=f"model file missing: {exc}") from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail="word prediction failed") from exc

	return WordPredictResponse(**result)


app.include_router(sentiment_router, prefix=settings.api_prefix)
app.include_router(spell_router, prefix=settings.api_prefix)
app.include_router(word_predict_router, prefix=settings.api_prefix)
