"""OpenAI-compatible API server for Gemma 3 4B via MLX."""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .engine import DEFAULT_MODEL, InferenceEngine
from .models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    ModelInfo,
    ModelListResponse,
    StreamChoice,
    ToolCall,
    FunctionCall,
    UsageInfo,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="MLX Server", description="OpenAI-compatible API for Gemma 3 4B")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return engine


def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    e = get_engine()
    return ModelListResponse(data=[ModelInfo(id=e.model_path)])


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    e = get_engine()

    # Convert pydantic messages to dicts
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    tools = None
    if request.tools:
        tools = [t.model_dump(exclude_none=True) for t in request.tools]

    prompt, images = e.build_prompt(messages, tools)

    stop = request.stop
    if isinstance(stop, str):
        stop = [stop]

    temperature = request.temperature if request.temperature is not None else 0.7
    top_p = request.top_p if request.top_p is not None else 0.9
    max_tokens = request.max_tokens if request.max_tokens is not None else 4096

    if request.stream:
        return EventSourceResponse(
            _stream_response(e, prompt, images, max_tokens, temperature, top_p, stop, tools, request.model),
            media_type="text/event-stream",
        )

    # Non-streaming
    text, prompt_tokens, completion_tokens = e.generate(
        prompt=prompt,
        images=images or None,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )

    # Check for tool calls in the response
    finish_reason = "stop"
    tool_calls_parsed = None
    if tools:
        clean_text, parsed = e.parse_tool_calls(text, tools)
        if parsed:
            tool_calls_parsed = [
                ToolCall(
                    index=i,
                    id=tc["id"],
                    type="function",
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for i, tc in enumerate(parsed)
            ]
            text = clean_text if clean_text else None
            finish_reason = "tool_calls"

    return ChatCompletionResponse(
        id=_make_id(),
        model=request.model,
        choices=[
            Choice(
                message=ChoiceMessage(
                    role="assistant",
                    content=text if not tool_calls_parsed else (text or None),
                    tool_calls=tool_calls_parsed,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_response(
    e: InferenceEngine,
    prompt: str,
    images: list[str] | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    tools: list[dict] | None,
    model_name: str,
):
    request_id = _make_id()
    created = int(time.time())

    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model_name,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
    )
    yield {"data": initial_chunk.model_dump_json()}

    full_text = ""
    prompt_tokens = 0
    gen_tokens = 0

    for token_text, is_final, pt, gt in e.stream_generate(
        prompt=prompt,
        images=images or None,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    ):
        prompt_tokens = pt
        gen_tokens = gt
        full_text += token_text

        if not is_final and token_text:
            chunk = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=model_name,
                choices=[StreamChoice(delta=DeltaMessage(content=token_text))],
            )
            yield {"data": chunk.model_dump_json()}

    # Check for tool calls in complete response
    finish_reason = "stop"
    if tools:
        clean_text, parsed = e.parse_tool_calls(full_text, tools)
        if parsed:
            finish_reason = "tool_calls"
            # Emit tool call chunks
            for i, tc in enumerate(parsed):
                tc_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=model_name,
                    choices=[
                        StreamChoice(
                            delta=DeltaMessage(
                                tool_calls=[
                                    ToolCall(
                                        index=i,
                                        id=tc["id"],
                                        type="function",
                                        function=FunctionCall(
                                            name=tc["function"]["name"],
                                            arguments=tc["function"]["arguments"],
                                        ),
                                    )
                                ]
                            )
                        )
                    ],
                )
                yield {"data": tc_chunk.model_dump_json()}

    # Final chunk with finish reason and usage
    final_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model_name,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=gen_tokens,
            total_tokens=prompt_tokens + gen_tokens,
        ),
    )
    yield {"data": final_chunk.model_dump_json()}
    yield {"data": "[DONE]"}


# ------------------------------------------------------------------
# Health / utility
# ------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MLX Server – OpenAI-compatible API")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model path")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--log-level", type=str, default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    global engine
    engine = InferenceEngine(model_path=args.model)
    engine.load()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
