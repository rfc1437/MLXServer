"""OpenAI-compatible API server for local LLMs (Gemma 3, Qwen3, …) via MLX."""

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

from .engine import DEFAULT_MODEL, InferenceEngine, ModelManager
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

app = FastAPI(title="MLX Server", description="OpenAI-compatible API for local LLMs on Apple Silicon")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager: ModelManager | None = None

# Number of recent messages to always preserve when summarizing
_KEEP_RECENT = 6


def get_engine(requested_model: str | None = None):
    if manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return manager.get_engine(requested_model)


def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


# ------------------------------------------------------------------
# Context window management
# ------------------------------------------------------------------


def _manage_context(
    e: InferenceEngine,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
) -> list[dict]:
    """Check if messages fit in the context window; summarize if needed.

    Returns the (possibly summarized) message list. Raises HTTPException
    with an OpenAI-compatible error if the conversation cannot fit.
    """
    context_length = e.context_length
    if context_length <= 0:
        return messages  # unknown context size, skip check

    prompt, _ = e.build_prompt(messages, tools)
    prompt_tokens = e.count_tokens(prompt)
    available = context_length - max_tokens

    if prompt_tokens <= available:
        return messages

    # --- Need to summarize ---
    logger.info(
        "Context window pressure: %d prompt tokens + %d max_tokens = %d "
        "(limit %d). Attempting summarization.",
        prompt_tokens, max_tokens, prompt_tokens + max_tokens, context_length,
    )

    # Split messages: system | middle (summarizable) | recent (kept)
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= _KEEP_RECENT:
        _raise_context_exceeded(prompt_tokens, max_tokens, context_length)

    recent = non_system[-_KEEP_RECENT:]
    middle = non_system[:-_KEEP_RECENT]

    # Generate summary of the middle messages
    summary_text = e.summarize_messages(middle)

    summary_msg = {
        "role": "user",
        "content": f"[Summary of earlier conversation]\n{summary_text}",
    }
    ack_msg = {
        "role": "assistant",
        "content": "Understood, I have the context from our earlier conversation.",
    }
    new_messages = system_msgs + [summary_msg, ack_msg] + recent

    # Re-check fit
    new_prompt, _ = e.build_prompt(new_messages, tools)
    new_prompt_tokens = e.count_tokens(new_prompt)

    if new_prompt_tokens + max_tokens > context_length:
        logger.warning(
            "Still over context limit after summarization: %d + %d = %d (limit %d)",
            new_prompt_tokens, max_tokens, new_prompt_tokens + max_tokens, context_length,
        )
        _raise_context_exceeded(new_prompt_tokens, max_tokens, context_length)

    logger.info(
        "Summarization reduced prompt from %d to %d tokens (saved %d).",
        prompt_tokens, new_prompt_tokens, prompt_tokens - new_prompt_tokens,
    )
    return new_messages


def _raise_context_exceeded(prompt_tokens: int, max_tokens: int, context_length: int):
    """Raise an OpenAI-compatible context_length_exceeded error."""
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": (
                    f"This model's maximum context length is {context_length} tokens. "
                    f"However, your messages resulted in {prompt_tokens} tokens and "
                    f"{max_tokens} tokens were requested for the completion "
                    f"({prompt_tokens + max_tokens} total). "
                    f"Please reduce the length of the messages or completion."
                ),
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        },
    )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    if manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return ModelListResponse(
        data=[
            ModelInfo(
                id=model_id,
                context_window=manager.get_context_length(model_id),
            )
            for model_id in manager.available_models
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    e = get_engine(request.model)

    # Convert pydantic messages to dicts
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    tools = None
    if request.tools:
        tools = [t.model_dump(exclude_none=True) for t in request.tools]

    stop = request.stop
    if isinstance(stop, str):
        stop = [stop]

    temperature = request.temperature if request.temperature is not None else 0.7
    top_p = request.top_p if request.top_p is not None else 0.9
    max_tokens = request.max_tokens if request.max_tokens is not None else 4096

    # Context window management: summarize if needed, error if impossible
    messages = _manage_context(e, messages, tools, max_tokens)

    prompt, images = e.build_prompt(messages, tools)

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
    e,
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

    # When tools are available we must buffer the full response before
    # emitting content — otherwise raw tool-call markup (```tool_code```
    # or <tool_call>) leaks into the streamed text.
    buffer_for_tools = bool(tools)

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

        if not buffer_for_tools and not is_final and token_text:
            chunk = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=model_name,
                choices=[StreamChoice(delta=DeltaMessage(content=token_text))],
            )
            yield {"data": chunk.model_dump_json()}

    # --- Post-generation: parse tool calls and emit clean content ------
    finish_reason = "stop"
    tool_calls_parsed = []

    if tools:
        clean_text, parsed = e.parse_tool_calls(full_text, tools)
        if parsed:
            finish_reason = "tool_calls"
            tool_calls_parsed = parsed
            full_text = clean_text or ""

    # Emit buffered content (when tools were present, this is the cleaned
    # text with tool-call markup stripped out)
    if buffer_for_tools and full_text.strip():
        content_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model_name,
            choices=[StreamChoice(delta=DeltaMessage(content=full_text))],
        )
        yield {"data": content_chunk.model_dump_json()}

    # Emit tool call chunks
    for i, tc in enumerate(tool_calls_parsed):
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

    global manager
    manager = ModelManager(default_model=args.model)
    manager.preload(args.model)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
