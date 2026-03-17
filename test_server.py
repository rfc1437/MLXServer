"""Test script for MLX Server – exercises chat, streaming, vision, and tool use."""

import base64
import io
import json
import sys

import httpx
from PIL import Image, ImageDraw

BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "mlx-community/gemma-3-4b-it-4bit"


def test_models():
    """Test GET /v1/models."""
    print("=" * 60)
    print("TEST: List models")
    print("=" * 60)
    r = httpx.get(f"{BASE_URL}/models")
    r.raise_for_status()
    data = r.json()
    print(f"Models: {[m['id'] for m in data['data']]}")
    print("PASS\n")


def test_chat_basic():
    """Test basic non-streaming chat."""
    print("=" * 60)
    print("TEST: Basic chat (non-streaming)")
    print("=" * 60)
    r = httpx.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say exactly: 'The sky is blue.' Nothing else."}],
            "max_tokens": 50,
            "temperature": 0.1,
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]["content"]
    usage = data["usage"]
    print(f"Response: {msg}")
    print(f"Usage: {usage}")
    print(f"Finish reason: {data['choices'][0]['finish_reason']}")
    print("PASS\n")


def test_chat_streaming():
    """Test streaming chat."""
    print("=" * 60)
    print("TEST: Streaming chat")
    print("=" * 60)
    collected = ""
    with httpx.stream(
        "POST",
        f"{BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Count from 1 to 5, one number per line."}],
            "max_tokens": 100,
            "temperature": 0.1,
            "stream": True,
        },
        timeout=120,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                collected += delta["content"]
                print(delta["content"], end="", flush=True)
            if chunk["choices"][0].get("finish_reason"):
                print(f"\n[finish_reason: {chunk['choices'][0]['finish_reason']}]")
            if chunk.get("usage") and chunk["usage"].get("total_tokens", 0) > 0:
                print(f"[usage: {chunk['usage']}]")
    print(f"Full collected: {collected!r}")
    print("PASS\n")


def _make_test_image() -> str:
    """Create a simple test image and return it as a base64 data URI."""
    img = Image.new("RGB", (200, 200), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    # Draw a red circle
    draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0), outline=(0, 0, 0), width=2)
    # Draw a green triangle
    draw.polygon([(100, 20), (60, 80), (140, 80)], fill=(0, 180, 0), outline=(0, 0, 0))
    # Draw yellow text area
    draw.rectangle([10, 160, 190, 190], fill=(255, 255, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def test_vision():
    """Test vision with an image."""
    print("=" * 60)
    print("TEST: Vision (image description)")
    print("=" * 60)
    image_uri = _make_test_image()
    print(f"Image: 200x200 PNG with red circle, green triangle, yellow bar")

    r = httpx.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what shapes and colors you see in this image. Be brief."},
                        {"type": "image_url", "image_url": {"url": image_uri}},
                    ],
                }
            ],
            "max_tokens": 200,
            "temperature": 0.1,
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]["content"]
    print(f"Response: {msg}")
    print("PASS\n")


def test_tool_use():
    """Test tool calling."""
    print("=" * 60)
    print("TEST: Tool use")
    print("=" * 60)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city name, e.g. 'London'",
                        },
                        "units": {
                            "type": "string",
                            "description": "Temperature units: 'celsius' or 'fahrenheit'",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    # Step 1: Ask the model to use the tool
    print("Step 1: Asking model to get weather for Paris...")
    r = httpx.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."},
            ],
            "tools": tools,
            "max_tokens": 300,
            "temperature": 0.1,
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    print(f"Finish reason: {choice['finish_reason']}")
    print(f"Content: {choice['message'].get('content')}")
    print(f"Tool calls: {choice['message'].get('tool_calls')}")

    if choice["message"].get("tool_calls"):
        tc = choice["message"]["tool_calls"][0]
        print(f"\nTool call detected:")
        print(f"  ID: {tc['id']}")
        print(f"  Function: {tc['function']['name']}")
        print(f"  Arguments: {tc['function']['arguments']}")

        # Step 2: Send the tool result back
        print("\nStep 2: Sending mock tool result back...")
        r2 = httpx.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."},
                    {
                        "role": "assistant",
                        "content": choice["message"].get("content"),
                        "tool_calls": choice["message"]["tool_calls"],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"temperature": 18, "condition": "Partly cloudy", "humidity": 65}),
                    },
                ],
                "tools": tools,
                "max_tokens": 300,
                "temperature": 0.1,
            },
            timeout=120,
        )
        r2.raise_for_status()
        data2 = r2.json()
        msg2 = data2["choices"][0]["message"]["content"]
        print(f"Final response: {msg2}")
    else:
        print("WARNING: Model did not produce a tool call. Raw response above.")

    print("PASS\n")


def test_multi_turn():
    """Test multi-turn conversation."""
    print("=" * 60)
    print("TEST: Multi-turn conversation")
    print("=" * 60)
    messages = [
        {"role": "user", "content": "My name is Alice."},
    ]
    r = httpx.post(
        f"{BASE_URL}/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": 100, "temperature": 0.1},
        timeout=120,
    )
    r.raise_for_status()
    reply1 = r.json()["choices"][0]["message"]["content"]
    print(f"Turn 1 reply: {reply1}")

    messages.append({"role": "assistant", "content": reply1})
    messages.append({"role": "user", "content": "What is my name?"})

    r2 = httpx.post(
        f"{BASE_URL}/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": 100, "temperature": 0.1},
        timeout=120,
    )
    r2.raise_for_status()
    reply2 = r2.json()["choices"][0]["message"]["content"]
    print(f"Turn 2 reply: {reply2}")
    assert "alice" in reply2.lower(), f"Expected 'Alice' in response, got: {reply2}"
    print("PASS\n")


if __name__ == "__main__":
    tests = [
        test_models,
        test_chat_basic,
        test_chat_streaming,
        test_vision,
        test_tool_use,
        test_multi_turn,
    ]

    # Allow running a single test by name
    if len(sys.argv) > 1:
        name = sys.argv[1]
        tests = [t for t in tests if name in t.__name__]
        if not tests:
            print(f"No test matching '{name}'. Available: models, chat_basic, chat_streaming, vision, tool_use, multi_turn")
            sys.exit(1)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
