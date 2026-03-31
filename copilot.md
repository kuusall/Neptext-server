# copilot.md

## Project Overview

This project is a FastAPI-based REST API server for Nepali text processing. It provides endpoints for sentiment analysis, spell correction (rule-based), and word prediction. The server accepts text containing Nepali unicode, emojis, romanized Nepali, and numbers, routes requests through an API gateway, logs all requests, and invokes the appropriate model for each task.

---

## Implementation Plan

### 1. Model & Service Preparation
- Use models in `models/` for sentiment analysis, spell correction, and word prediction.
- Only use `old/` directory for reference, not for direct integration.
- Create a service layer in `services/` for each task, handling preprocessing, model invocation, and postprocessing.

### 2. API Server Implementation
- Build the FastAPI app in `app.py`.
- Add routers for `/sentiment`, `/spell-correct`, and `/word-predict`.
- Implement request/response logging middleware.
- Enable CORS for frontend access.
- Serve OpenAPI docs.

### 3. API Schemas
- Define Pydantic models for request/response validation.

### 4. Logging & Gateway
- Log all requests and responses with relevant metadata.
- FastAPI will act as the API gateway unless otherwise specified.

### 5. Documentation & Requirements
- Write `requirements.txt` for dependencies.
- Create `curl.md` for API documentation with request/response examples.

### 6. Testing & Verification
- Test endpoints with unicode, emojis, romanized Nepali, and numbers.
- Verify OpenAPI docs and `curl.md` for accuracy.

---

## Key Files

- `app.py`: FastAPI app, routers, middleware, logging, docs
- `models/`: Model wrappers for each task
- `services/`: Service layer for business logic
- `model_dicts/`: Model weights/resources
- `config/env.py`: Configuration
- `requirements.txt`: Python dependencies
- `curl.md`: API documentation

---

## Decisions & Notes

- `old/` is for reference only.
- Standard error responses will be defined for invalid input/model errors.
- Clarify if romanized Nepali requires transliteration.
- Logging destination (file/stdout) to be finalized.

---

This file should be updated as the project evolves and decisions are finalized.
