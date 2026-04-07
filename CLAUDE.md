# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**XinYu (心语)** is a mental health emotional-support chat assistant thesis project. It combines an NLP analysis module (emotion/intent/intensity classification) with an LLM orchestration layer to provide safe, risk-aware conversations with a counselor dashboard.

The project follows a strict **module plan governance model**: no code may be written for a module until an approved plan document exists in `/plan/`. Read `PLANS.md` before starting any new module. Check `/plan/` for the current module status.

## Development Commands

Dependencies are managed with **uv** (`pyproject.toml` at the project root).  
All `uv run` commands are executed from the **project root** (`D:\PycharmProjects\XinYu\`).

```bash
# First-time setup — create .venv and install all deps
uv sync              # runtime + dev deps
uv sync --group nlp  # also install NLP training deps (torch, transformers, …)

# Run the dev server  ← MUST run from project root, not backend/
uv run uvicorn backend.app.main:app --reload --port 8000

# Run all tests
uv run pytest

# Run a single test file
uv run pytest backend/tests/unit/test_context_prompt_services.py -v

# Run a single test by name
uv run pytest backend/tests/unit/test_context_prompt_services.py::test_summary_present -v

# Apply database migrations
uv run alembic --config backend/alembic.ini upgrade head

# Create a new migration
uv run alembic --config backend/alembic.ini revision --autogenerate -m "description"

# Add / remove a dependency
uv add <package>
uv remove <package>
```

**Health check:** `GET http://127.0.0.1:8000/api/v1/health`

## Architecture

### High-Level Stack
- **Backend:** FastAPI + SQLAlchemy 2.0 + Alembic + PostgreSQL 16
- **Config:** Pydantic-settings, env prefix `XINYU_` (e.g., `XINYU_DATABASE_URL`)
- **Default DB URL:** `postgresql+psycopg://postgres:postgres@127.0.0.1:5432/xinyu`
- **Frontend / NLP / ML:** Planned, not yet implemented

### Key Structural Patterns

**App Factory:** `backend/app/main.py` — `create_app()` returns the FastAPI instance used by uvicorn.

**API routing:** `app/api/router.py` → `app/api/v1/router.py` → individual endpoint files. All endpoints live under `/api/v1/`.

**Dependency injection:** `app/dependencies/services.py` defines `AppContainer` (a frozen dataclass). Routers receive services via FastAPI's `Depends()`. Never instantiate services directly in route handlers.

**Service layer:** Business logic lives in `app/services/`. Currently:
- `ContextService` — builds `ContextWindow` from DB (summaries + recent messages + risk level)
- `PromptService` — assembles `PromptBundle` (system prompt + history + current analysis)
- `NLPService` / `StubNLPService` / `RealNLPService` — emotion/intent/intensity analysis; `RealNLPService` loads a trained `MentalHealthMultiTaskModel` when `XINYU_NLP_MODEL_PATH` is set
- `RiskService` — pure risk evaluator; maps `AnalysisResult` → `RiskAssessment` (L0–L3)
- `ResourceService` — queries `resource_catalog` for crisis resources by risk level
- `LLMProvider` / `FakeLLMProvider` / `DoubaoLLMProvider` — streaming LLM adapter
- `ChatService` — full turn orchestrator; yields `StreamEvent` async generator

**Shared keyword lists:** `app/core/crisis_keywords.py` — `L3_KEYWORDS` and `L2_KEYWORDS` frozensets imported by both `RiskService` and `RealNLPService`.

**Settings:** `app/core/config.py` — `get_settings()` is `@lru_cache`-decorated; call it via `Depends(get_settings)` in routes.

### Data Flow (Chat, Planned)
```
POST /api/v1/chat/{session_id}/stream
  → NLPService → AnalysisResult (emotion, intent, intensity, risk_score)
  → RiskService → RiskAssessment (risk_level L0–L3, reasons)
  → ContextService → ContextWindow (summary + recent 6–10 msgs)
  → PromptService → PromptBundle (system prompt + history + analysis)
  → LLMProvider.stream_reply() → SSE events: meta → [alert] → token → complete
  → Persist to DB (chat_messages, message_analyses, alert_events)
```

### Risk Levels
- **L0** — Normal conversation
- **L1** — Elevated concern (monitor)
- **L2** — Warning (explicit concerning content)
- **L3** — Emergency (crisis / self-harm)

### Database Tables (all created by `20260309_01_initial_core_tables.py`)
`visitor_profiles`, `chat_sessions`, `chat_messages`, `message_analyses`, `conversation_summaries`, `alert_events`, `resource_catalog`, `counselor_accounts`

### Context Continuity (3-Layer)
1. **Fact layer** — raw DB tables
2. **Compression layer** — `conversation_summaries` (refreshed every 4 msgs after the first 10)
3. **Runtime window** — `PromptBundle` sent to LLM (recent 6–10 messages)

## Module Status
| Module | Plan File | Status |
|--------|-----------|--------|
| 01 Backend Foundation | `plan/01-backend-foundation.md` | ✅ Complete |
| 02 Database & Models | `plan/02-database-and-models.md` | ✅ Complete |
| 03 Context & Prompt | `plan/03-context-and-prompt.md` | ✅ Complete |
| 04 LLM & Risk | `plan/04-llm-and-risk.md` | ✅ Complete |
| 05 NLP Pipeline | `plan/05-nlp-pipeline.md` | ✅ Complete |
| 06 Chat Endpoint | `plan/06-chat-endpoint.md` | ✅ Complete |
| 07 Frontend | `plan/07-frontend.md` | ✅ Complete |

## Governance Rules (from PLANS.md)
- Every module needs an approved plan document before any code is written.
- Plans are living documents — update them as work proceeds (surprises, decisions, outcomes).
- Record all significant decisions with rationale and date in the plan's Decision Log.
- Plans must be self-contained (assume no external context).
