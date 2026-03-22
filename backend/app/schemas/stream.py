from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class MetaEventPayload(BaseModel):
    emotion: str
    intent: str
    intensity: float
    risk_level: str


class TokenEventPayload(BaseModel):
    text: str


class AlertEventPayload(BaseModel):
    risk_level: str
    resources: list[dict[str, str]] = Field(default_factory=list)


class CompleteEventPayload(BaseModel):
    message_id: str
    latency_ms: int


class MetaEvent(BaseModel):
    event: Literal["meta"] = "meta"
    data: MetaEventPayload


class TokenEvent(BaseModel):
    event: Literal["token"] = "token"
    data: TokenEventPayload


class AlertEvent(BaseModel):
    event: Literal["alert"] = "alert"
    data: AlertEventPayload


class CompleteEvent(BaseModel):
    event: Literal["complete"] = "complete"
    data: CompleteEventPayload


# Discriminated union for type-safe event handling across the service boundary.
StreamEvent = Annotated[
    Union[MetaEvent, TokenEvent, AlertEvent, CompleteEvent],
    Field(discriminator="event"),
]
