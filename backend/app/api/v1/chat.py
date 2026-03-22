from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.app.dependencies.services import AppContainer, get_container
from backend.app.models.chat_session import ChatSession

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


@router.post("/{session_id}/stream")
async def stream_chat(
    session_id: str,
    body: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> EventSourceResponse:
    """Stream a chat turn as Server-Sent Events.

    Event order (normal): meta → token… → complete
    Event order (high-risk): meta → alert → token… → complete
    """
    # Pre-flight: verify session before opening the SSE connection.
    # Once EventSourceResponse starts, headers are already sent — a 404 inside
    # the generator cannot be surfaced as an HTTP status code.
    with container.session_factory() as db:
        if db.get(ChatSession, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator() -> AsyncGenerator[dict, None]:
        async for evt in container.chat_service.stream_chat(session_id, body.message):
            yield {"event": evt.event, "data": evt.data.model_dump_json()}

    return EventSourceResponse(event_generator(), media_type="text/event-stream")
