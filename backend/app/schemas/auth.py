from __future__ import annotations

from pydantic import BaseModel, Field


class VisitorRegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    display_name: str | None = Field(None, max_length=100)
    real_name: str = Field(..., min_length=1, max_length=100)
    college: str = Field(..., max_length=100)
    student_id: str = Field(..., min_length=1, max_length=50)


class VisitorLoginRequest(BaseModel):
    username: str
    password: str


class CounselorLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: str


class MeResponse(BaseModel):
    user_id: str
    role: str
    username: str
    display_name: str | None = None
    real_name: str | None = None
    college: str | None = None
    student_id: str | None = None
    is_guest: bool = False
