"""Model loading and inference engine using mlx_vlm (supports both text and vision)."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path

import mlx.core as mx
import mlx_vlm
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/gemma-3-4b-it-4bit"


# ------------------------------------------------------------------
# Helpers for Gemma 3 tool_code format
# ------------------------------------------------------------------

_JSON_TO_PYTHON_TYPE = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}

_JSON_TYPE_DEFAULTS = {
    "string": '""',
    "integer": "0",
    "number": "0.0",
    "boolean": "False",
    "array": "[]",
    "object": "{}",
}


def _json_type_to_python(json_type: str) -> str:
    return _JSON_TO_PYTHON_TYPE.get(json_type, "str")


def _json_type_default(json_type: str) -> str:
    return _JSON_TYPE_DEFAULTS.get(json_type, "None")


def _python_repr(value) -> str:
    """Produce a Python-repr-style string for a value."""
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(value)


def _parse_python_call(call_str: str, tool_defs: dict[str, dict] | None = None) -> tuple[str, dict]:
    """Parse a function call string into (name, args_dict).

    Handles multiple formats:
    1. Python-style: func_name(arg1="value1", arg2=42)
    2. Shell-style:  func_name arg1 arg2  (common with small LLMs)
    3. Mixed:        func_name("value")   (positional args)

    tool_defs maps function names to their parameter schemas, used to
    infer which parameter a positional/shell-style argument maps to.
    """
    import ast

    call_str = call_str.strip()

    # Try Python-style: function_name(...)
    m = re.match(r"(\w+)\s*\((.*)\)\s*$", call_str, re.DOTALL)
    if m:
        name = m.group(1)
        args_str = m.group(2).strip()

        if not args_str:
            return name, {}

        # Try parsing as a Python function call via dict()
        try:
            tree = ast.parse(f"dict({args_str})", mode="eval")
            call_node = tree.body
            args = {}
            # Handle keyword arguments: func(key="val")
            for kw in call_node.keywords:
                args[kw.arg] = ast.literal_eval(kw.value)
            # Handle positional arguments: func("val1", "val2")
            if call_node.args and not args:
                param_names = _get_param_names(name, tool_defs)
                for i, arg_node in enumerate(call_node.args):
                    val = ast.literal_eval(arg_node)
                    if i < len(param_names):
                        args[param_names[i]] = val
                    else:
                        args[f"arg{i}"] = val
            if args:
                return name, args
        except Exception:
            pass

        # Fallback: regex-based key=value parsing
        args = {}
        for pair_match in re.finditer(r"(\w+)\s*=\s*(.+?)(?:,\s*(?=\w+\s*=)|$)", args_str, re.DOTALL):
            key = pair_match.group(1)
            val_str = pair_match.group(2).strip()
            try:
                args[key] = ast.literal_eval(val_str)
            except Exception:
                args[key] = val_str
        return name, args

    # Shell-style: "func_name arg1 arg2" or "func_name some/path"
    # Also handles: "func_name -flag arg" (common with shell tools)
    parts = call_str.split(None, 1)
    if parts and re.match(r"^\w+$", parts[0]):
        name = parts[0]
        if len(parts) == 1:
            return name, {}

        rest = parts[1].strip()
        param_names = _get_param_names(name, tool_defs)
        first_param = param_names[0] if param_names else "input"
        return name, {first_param: rest}

    # Last resort: treat the entire block as a command for the first
    # known tool that looks like a shell/command tool, or just fail
    raise ValueError(f"Cannot parse as function call: {call_str!r}")


def _get_param_names(func_name: str, tool_defs: dict[str, dict] | None) -> list[str]:
    """Get ordered parameter names for a function from tool definitions."""
    if not tool_defs or func_name not in tool_defs:
        return []
    params = tool_defs[func_name].get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])
    # Required params first, then optional
    optional = [k for k in properties if k not in required]
    return list(required) + optional


class PromptCache:
    """Manages KV cache reuse across requests with shared prompt prefixes.

    Gemma 3 uses a mix of KVCache (full attention every 6th layer) and
    RotatingKVCache (sliding window, 1024 tokens). Since RotatingKVCache
    cannot be safely trimmed mid-sequence, we only reuse the cache when
    the ENTIRE cached token sequence is a prefix of the new prompt.

    In multi-turn chat this is the common case: each new request extends
    the previous prompt with the assistant response + new user message.
    """

    def __init__(self):
        self._cache: list | None = None
        self._cached_token_ids: list[int] | None = None

    def get_reusable_length(self, new_token_ids: list[int]) -> int:
        """Return cached length if the entire cache is a valid prefix, else 0."""
        if self._cached_token_ids is None or self._cache is None:
            return 0
        cached_len = len(self._cached_token_ids)
        if cached_len > len(new_token_ids):
            return 0
        for i in range(cached_len):
            if self._cached_token_ids[i] != new_token_ids[i]:
                return 0
        return cached_len

    def update(self, cache: list, token_ids: list[int]) -> None:
        """Store cache and the token IDs it was built from."""
        self._cache = cache
        self._cached_token_ids = list(token_ids)

    def clear(self) -> None:
        self._cache = None
        self._cached_token_ids = None

    @property
    def cache(self):
        return self._cache


class InferenceEngine:
    """Manages model loading and text/vision generation."""

    def __init__(self, model_path: str = DEFAULT_MODEL):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.config = None
        self._lock = threading.Lock()
        self._prompt_cache = PromptCache()

    def load(self) -> None:
        logger.info("Loading model %s ...", self.model_path)
        self.model, self.processor = mlx_vlm.load(self.model_path)
        # Load model config for chat template
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_image_url(url: str) -> str:
        """Convert a data URI or URL to a file path that mlx_vlm can consume."""
        if url.startswith("data:"):
            # data:image/png;base64,iVBOR...
            header, b64data = url.split(",", 1)
            img_bytes = base64.b64decode(b64data)
            img = Image.open(io.BytesIO(img_bytes))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp, format="PNG")
            tmp.close()
            return tmp.name
        # Assume it's a URL or local path – mlx_vlm handles URLs natively
        return url

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> tuple[str, list[str]]:
        """Build a prompt string and collect image paths from messages.

        Returns (prompt_str, image_paths).
        """
        image_paths: list[str] = []
        formatted_messages: list[dict] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "system":
                text = self._get_text_content(content)
                # Inject tool definitions into system prompt
                if tools:
                    text = self._inject_tools_into_system(text, tools)
                formatted_messages.append({"role": "user", "content": text})
                # Gemma 3 doesn't have a system role; we use the user role
                # and add a model acknowledgment
                formatted_messages.append({
                    "role": "assistant",
                    "content": "Understood. I will follow these instructions.",
                })
            elif role == "user":
                text, imgs = self._extract_content_parts(content)
                image_paths.extend(imgs)
                formatted_messages.append({"role": "user", "content": text})
            elif role == "assistant":
                text = self._get_text_content(content) or ""
                if tool_calls:
                    # Format tool calls in the way Gemma 3 expects
                    tc_text = self._format_tool_calls_for_prompt(tool_calls)
                    text = (text + "\n" + tc_text).strip()
                formatted_messages.append({"role": "assistant", "content": text})
            elif role == "tool":
                # Tool results use Gemma 3's tool_output format
                tool_text = self._get_text_content(content) or ""
                result_msg = f"```tool_output\n{tool_text}\n```"
                formatted_messages.append({"role": "user", "content": result_msg})

        # If the first system prompt had no tools but we have tools, inject at start
        if tools and not any(m.get("role") == "system" for m in messages):
            tool_system = self._build_tool_system_prompt(tools)
            formatted_messages.insert(0, {"role": "user", "content": tool_system})
            formatted_messages.insert(1, {
                "role": "assistant",
                "content": "Understood. I will follow these instructions and use tools when appropriate.",
            })

        # Gemma 3 requires strictly alternating user/assistant turns.
        # Merge consecutive same-role messages and ensure it starts with user.
        formatted_messages = self._merge_consecutive_roles(formatted_messages)

        # Apply chat template via mlx_vlm
        prompt = mlx_vlm.apply_chat_template(
            self.processor,
            self.config,
            formatted_messages,
            add_generation_prompt=True,
            num_images=len(image_paths),
        )

        return prompt, image_paths

    @staticmethod
    def _merge_consecutive_roles(messages: list[dict]) -> list[dict]:
        """Merge consecutive messages with the same role into one.

        Gemma 3's chat template enforces strict user/assistant alternation.
        """
        if not messages:
            return messages

        merged = [messages[0].copy()]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                # Merge content with the previous message
                merged[-1]["content"] = (
                    merged[-1].get("content", "") + "\n\n" + msg.get("content", "")
                )
            else:
                merged.append(msg.copy())

        # Ensure conversation starts with user
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": ""})

        return merged

    def _get_text_content(self, content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # list of content parts
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part["text"])
        return "\n".join(parts)

    def _extract_content_parts(self, content) -> tuple[str, list[str]]:
        """Extract text and image paths from content parts."""
        if isinstance(content, str):
            return content, []
        if content is None:
            return "", []

        texts = []
        images = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part["text"])
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    images.append(self._decode_image_url(url))
        return "\n".join(texts), images

    def _inject_tools_into_system(self, system_text: str, tools: list[dict]) -> str:
        tool_block = self._build_tool_system_prompt(tools)
        return f"{system_text}\n\n{tool_block}"

    def _build_tool_system_prompt(self, tools: list[dict]) -> str:
        """Build the tool system prompt using Google's official Gemma 3 format.

        Uses the tool_code/tool_output convention recommended by Google:
        - Tools defined as Python function signatures with docstrings
        - Model outputs calls in ```tool_code``` fenced blocks
        - Results returned in ```tool_output``` fenced blocks
        """
        func_defs = []
        for tool in tools:
            func = tool.get("function", tool)
            func_defs.append(self._tool_to_python_signature(func))

        functions_block = "\n\n".join(func_defs)

        return (
            "At each turn, if you decide to invoke any of the function(s), "
            "it should be wrapped with ```tool_code```. "
            "The python methods described below are imported and available, "
            "you can only use defined methods. "
            "The generated code should be readable and efficient. "
            "The response to a method will be wrapped in ```tool_output``` "
            "use it to call more tools or generate a helpful, friendly response.\n"
            "\n"
            f"{functions_block}"
        )

    @staticmethod
    def _tool_to_python_signature(func: dict) -> str:
        """Convert an OpenAI function definition to a Python function signature with docstring."""
        name = func["name"]
        desc = func.get("description", "")
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))

        # Build parameter list
        param_parts = []
        doc_args = []
        for pname, pinfo in properties.items():
            ptype = _json_type_to_python(pinfo.get("type", "str"))
            pdesc = pinfo.get("description", "")
            if pname in required:
                param_parts.append(f"{pname}: {ptype}")
            else:
                default = _json_type_default(pinfo.get("type", "str"))
                param_parts.append(f"{pname}: {ptype} = {default}")
            doc_args.append(f"      {pname}: {pdesc}" if pdesc else f"      {pname}")

        sig = f"def {name}({', '.join(param_parts)}):"
        doc_lines = [f'    """{desc}']
        if doc_args:
            doc_lines.append("")
            doc_lines.append("    Args:")
            doc_lines.extend(doc_args)
        doc_lines.append('    """')

        return sig + "\n" + "\n".join(doc_lines)

    def _format_tool_calls_for_prompt(self, tool_calls: list[dict]) -> str:
        """Format OpenAI-style tool calls back into Gemma 3 tool_code blocks."""
        parts = []
        for tc in tool_calls:
            func = tc.get("function", tc)
            name = func["name"]
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            # Format as Python function call
            arg_parts = [f"{k}={_python_repr(v)}" for k, v in args.items()]
            call_str = f"{name}({', '.join(arg_parts)})"
            parts.append(f"```tool_code\n{call_str}\n```")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Prefix cache & generation
    # ------------------------------------------------------------------

    # Common kwargs for mlx_vlm generate calls
    # Note: KV cache quantization is not supported with Gemma 3's RotatingKVCache
    _GENERATE_KWARGS: dict = {}

    def _get_tokenizer(self):
        """Get the underlying tokenizer from the processor."""
        proc = self.processor
        return proc.tokenizer if hasattr(proc, "tokenizer") else proc

    def _prepare_generation(
        self, prompt: str, images: list[str] | None = None
    ) -> dict:
        """Tokenize prompt, check prefix cache, return generation kwargs.

        Returns a dict with keys:
          input_ids, pixel_values, mask, prompt_cache,
          _full_token_ids, _prompt_token_count
        """
        from mlx_vlm.models import cache as cache_module
        from mlx_vlm.utils import prepare_inputs

        model_type = getattr(self.config, "model_type", "")
        add_special_tokens = (
            not hasattr(self.processor, "chat_template")
            if model_type in ("gemma3", "gemma3n")
            else True
        )
        image_token_index = getattr(self.model.config, "image_token_index", None)

        # Tokenize the full prompt (+ process pixel values if images present)
        inputs = prepare_inputs(
            self.processor,
            images=images if images else None,
            prompts=prompt,
            image_token_index=image_token_index,
            add_special_tokens=add_special_tokens,
        )
        full_input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        mask = inputs.get("attention_mask")

        full_token_list = full_input_ids.flatten().tolist()
        prefix_len = self._prompt_cache.get_reusable_length(full_token_list)

        if prefix_len > 0:
            suffix_token_list = full_token_list[prefix_len:]

            # If the suffix contains image placeholder tokens, we can't skip
            # the vision encoder — fall back to full processing.
            if (
                image_token_index is not None
                and image_token_index in suffix_token_list
            ):
                logger.info(
                    "New images in suffix — prefix cache invalidated"
                )
                prefix_len = 0

        if prefix_len > 0:
            suffix_ids = mx.array([suffix_token_list])
            logger.info(
                "Prefix cache hit: reusing %d/%d tokens (%.1f%%), "
                "processing %d new tokens",
                prefix_len,
                len(full_token_list),
                100 * prefix_len / len(full_token_list),
                len(suffix_token_list),
            )
            return {
                "input_ids": suffix_ids,
                "pixel_values": None,  # images already in cached KV
                "mask": None,
                "prompt_cache": self._prompt_cache.cache,
                "_full_token_ids": full_token_list,
                "_prompt_token_count": len(full_token_list),
            }

        # Cache miss — create a fresh KV cache
        cache = cache_module.make_prompt_cache(self.model.language_model)
        logger.info(
            "Prefix cache miss: processing %d tokens from scratch",
            len(full_token_list),
        )
        return {
            "input_ids": full_input_ids,
            "pixel_values": pixel_values,
            "mask": mask,
            "prompt_cache": cache,
            "_full_token_ids": full_token_list,
            "_prompt_token_count": len(full_token_list),
        }

    def _save_cache(self, prep: dict, generated_tokens: list[int]) -> None:
        """Persist the KV cache and token IDs after generation."""
        full_sequence = prep["_full_token_ids"] + generated_tokens
        self._prompt_cache.update(prep["prompt_cache"], full_sequence)

    def generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        repetition_penalty: float = 1.1,
    ) -> tuple[str, int, int]:
        """Generate a complete response. Returns (text, prompt_tokens, completion_tokens)."""
        with self._lock:
            prep = self._prepare_generation(prompt, images)
            prompt_token_count = prep["_prompt_token_count"]

            # Ensure stopping criteria is initialised
            tokenizer = self._get_tokenizer()
            tokenizer.stopping_criteria.reset(self.model.config.eos_token_id)

            text = ""
            generated_tokens: list[int] = []
            gen_tokens = 0

            for result in mlx_vlm.stream_generate(
                self.model,
                self.processor,
                prompt,
                input_ids=prep["input_ids"],
                pixel_values=prep.get("pixel_values"),
                mask=prep.get("mask"),
                prompt_cache=prep["prompt_cache"],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                **self._GENERATE_KWARGS,
            ):
                text += result.text
                if result.token is not None:
                    generated_tokens.append(result.token)
                gen_tokens = result.generation_tokens

            self._save_cache(prep, generated_tokens)

            if stop:
                text = self._apply_stop(text, stop)
            return text, prompt_token_count, gen_tokens

    def stream_generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        repetition_penalty: float = 1.1,
    ) -> Generator[tuple[str, bool, int, int], None, None]:
        """Stream tokens. Yields (token_text, is_final, prompt_tokens, gen_tokens)."""
        with self._lock:
            prep = self._prepare_generation(prompt, images)
            prompt_token_count = prep["_prompt_token_count"]

            # Ensure stopping criteria is initialised
            tokenizer = self._get_tokenizer()
            tokenizer.stopping_criteria.reset(self.model.config.eos_token_id)

            accumulated = ""
            generated_tokens: list[int] = []
            gen_tokens = 0

            try:
                for result in mlx_vlm.stream_generate(
                    self.model,
                    self.processor,
                    prompt,
                    input_ids=prep["input_ids"],
                    pixel_values=prep.get("pixel_values"),
                    mask=prep.get("mask"),
                    prompt_cache=prep["prompt_cache"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    **self._GENERATE_KWARGS,
                ):
                    token_text = result.text
                    accumulated += token_text
                    if result.token is not None:
                        generated_tokens.append(result.token)
                    gen_tokens = result.generation_tokens

                    if stop and self._check_stop(accumulated, stop):
                        trimmed = self._apply_stop(accumulated, stop)
                        safe_delta = trimmed[
                            len(accumulated) - len(token_text) :
                        ]
                        yield safe_delta, True, prompt_token_count, gen_tokens
                        return

                    yield token_text, False, prompt_token_count, gen_tokens

                # Final yield to signal completion
                yield "", True, prompt_token_count, gen_tokens
            finally:
                self._save_cache(prep, generated_tokens)

    @staticmethod
    def _apply_stop(text: str, stop: list[str]) -> str:
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                text = text[:idx]
        return text

    @staticmethod
    def _check_stop(text: str, stop: list[str]) -> bool:
        return any(s in text for s in stop)

    # ------------------------------------------------------------------
    # Tool call parsing from model output
    # ------------------------------------------------------------------

    @staticmethod
    def parse_tool_calls(
        text: str, tools: list[dict] | None = None
    ) -> tuple[str, list[dict]]:
        """Parse tool calls from model output using Gemma 3's tool_code format.

        Detects ```tool_code ... ``` blocks containing Python-style or
        shell-style function calls.

        Returns (clean_text, tool_calls) where tool_calls is a list of
        {"id": str, "type": "function", "function": {"name": str, "arguments": str}}.
        """
        # Build a lookup of function name -> parameter schema
        tool_defs: dict[str, dict] = {}
        if tools:
            for tool in tools:
                func = tool.get("function", tool)
                tool_defs[func["name"]] = func

        tool_calls = []
        pattern = r"```tool_code\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)

        clean_text = re.sub(r"```tool_code\s*.*?\s*```", "", text, flags=re.DOTALL).strip()

        for i, match in enumerate(matches):
            call_str = match.strip()
            try:
                name, args = _parse_python_call(call_str, tool_defs)
                tool_calls.append({
                    "id": f"call_{i}_{hash(call_str) % 10**8:08d}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                })
            except Exception as e:
                logger.warning("Failed to parse tool_code call %r: %s", call_str, e)

        return clean_text, tool_calls
