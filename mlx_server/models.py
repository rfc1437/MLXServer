"""OpenAI API compatible request/response models."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Request models ---


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    index: int = 0
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ContentPartText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageURL(BaseModel):
    url: str  # Can be a URL or base64 data URI
    detail: str | None = None


class ContentPartImage(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


ContentPart = ContentPartText | ContentPartImage


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "gemma-3-4b-it"
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    max_tokens: int | None = 4096
    stream: bool = False
    stop: str | list[str] | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    n: int | None = 1


# --- Response models ---


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: UsageInfo


# --- Streaming response models ---


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    usage: UsageInfo | None = None


# --- Model listing ---


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
