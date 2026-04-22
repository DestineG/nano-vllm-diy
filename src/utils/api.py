from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.config.sampling_params import SamplingParams
from src.engine.llm_engine import LLMEngine
from src.models.qwen3 import Qwen3ForCausalLM


DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[Any]
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float = Field(default=1.0, gt=1e-10)
    max_tokens: int = Field(default=512, ge=1)
    stream: bool = False
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    ignore_eos: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str] | list[list[int]]
    temperature: float = Field(default=1.0, gt=1e-10)
    max_tokens: int = Field(default=512, ge=1)
    stream: bool = False
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    ignore_eos: bool = False


class ResponseCreateRequest(BaseModel):
    model: str | None = None
    input: str | list[Any]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float = Field(default=1.0, gt=1e-10)
    max_output_tokens: int = Field(default=512, ge=1)
    stream: bool = False
    top_p: float | None = None
    user: str | None = None
    ignore_eos: bool = False


def _now() -> int:
    return int(time.time())


def _make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _normalize_text_content(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    if not content:
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if "text" in item:
                parts.append(str(item["text"]))
            elif "content" in item:
                parts.append(str(item["content"]))
        else:
            parts.append(str(item))
    return "\n".join(part for part in parts if part)


def _build_chat_prompt(engine: LLMEngine, messages: list[Message]) -> str:
    # Most chat templates don't support a separate developer role.
    # Map it to system semantics for compatibility with Codex requests.
    role_map = {"developer": "system"}
    chat_messages = [
        {
            "role": role_map.get(message.role, message.role),
            "content": _normalize_text_content(message.content),
        }
        for message in messages
    ]
    return engine.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)


def _messages_from_responses_input(input_data: str | list[Any]) -> list[Message]:
    if isinstance(input_data, str):
        return [Message(role="user", content=input_data)]

    messages: list[Message] = []
    for item in input_data:
        if isinstance(item, dict) and "role" in item and "content" in item:
            role = item.get("role", "user")
            if role not in {"system", "developer", "user", "assistant", "tool"}:
                role = "user"
            messages.append(Message(role=role, content=item["content"]))
            continue

        if isinstance(item, dict) and item.get("type") in {"message", "input_message"}:
            role = item.get("role", "user")
            if role not in {"system", "developer", "user", "assistant", "tool"}:
                role = "user"
            messages.append(Message(role=role, content=item.get("content", "")))
            continue

        if isinstance(item, dict) and item.get("type") in {"input_text", "text"}:
            messages.append(Message(role="user", content=item.get("text", "")))
            continue

        if isinstance(item, dict) and item.get("type") == "function_call_output":
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            messages.append(Message(role="tool", content=f"call_id={call_id}\n{output}"))
            continue

        messages.append(Message(role="user", content=str(item)))

    if not messages:
        messages.append(Message(role="user", content=""))
    return messages


def _build_responses_object(
    response_id: str,
    model: str,
    output_text: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    message_id = _make_id("msg")
    return {
        "id": response_id,
        "object": "response",
        "created_at": _now(),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": output_text,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _build_function_call_response_object(
    response_id: str,
    model: str,
    tool_name: str,
    arguments_json: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    item_id = _make_id("item")
    call_id = _make_id("call")
    return {
        "id": response_id,
        "object": "response",
        "created_at": _now(),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": tool_name,
                "arguments": arguments_json,
            }
        ],
        "output_text": "",
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _get_allowed_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    if not tools:
        return set()
    names: set[str] = set()
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _tool_prompt_suffix(tools: list[dict[str, Any]] | None, tool_choice: str | dict[str, Any] | None) -> str:
    if not tools:
        return ""

    funcs = []
    for tool in tools:
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            continue
        fn = tool["function"]
        funcs.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )

    required_note = ""
    if isinstance(tool_choice, str) and tool_choice == "required":
        required_note = "You MUST return a tool_call JSON object."
    if isinstance(tool_choice, dict) and isinstance(tool_choice.get("function"), dict):
        forced_name = tool_choice["function"].get("name")
        if forced_name:
            required_note = f"You MUST call this tool name exactly: {forced_name}."

    tool_catalog = json.dumps(funcs, ensure_ascii=True)
    return (
        "\n\nTool-calling mode:\n"
        "If a tool should be called, output EXACTLY one JSON object with this schema:\n"
        '{"type":"tool_call","name":"<tool_name>","arguments":{...}}\n'
        "If no tool is needed, output EXACTLY one JSON object:\n"
        '{"type":"message","content":"<assistant reply>"}\n'
        f"Available tools: {tool_catalog}\n"
        f"{required_note}"
    )


def _sse_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=True)}\n\n"


def _normalize_model_name(model_path: str) -> str:
    return os.path.expanduser(model_path).rstrip("/").split("/")[-1] or "model"


@lru_cache(maxsize=1)
def get_engine(model_path: str = DEFAULT_MODEL_PATH) -> LLMEngine:
    return LLMEngine(os.path.expanduser(model_path), Qwen3ForCausalLM)


async def _run_generate(engine: LLMEngine, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams]):
    return await asyncio.to_thread(engine.generate, prompts, sampling_params, False)


def _build_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def create_app(model_path: str = DEFAULT_MODEL_PATH) -> FastAPI:
    app = FastAPI(title="nano-vllm-diy OpenAI-compatible API", version="0.1.0")

    @app.on_event("startup")
    def _warmup() -> None:
        app.state.engine = get_engine(model_path)
        app.state.model_name = _normalize_model_name(model_path)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        model_name = app.state.model_name
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": _now(),
                    "owned_by": "nano-vllm-diy",
                }
            ],
        }

    @app.get("/v1/models/{model_id}")
    def get_model(model_id: str) -> dict[str, Any]:
        if model_id != app.state.model_name:
            raise HTTPException(status_code=404, detail="model not found")
        return {
            "id": model_id,
            "object": "model",
            "created": _now(),
            "owned_by": "nano-vllm-diy",
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> JSONResponse:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming is not implemented")

        engine = app.state.engine
        prompt = _build_chat_prompt(engine, request.messages)
        outputs = await _run_generate(
            engine,
            [prompt],
            SamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                ignore_eos=request.ignore_eos,
            ),
        )
        choice_text = outputs[0]["text"]
        prompt_token_count = len(engine.tokenizer.encode(prompt))
        completion_token_count = len(outputs[0]["token_ids"])
        response = {
            "id": _make_id("chatcmpl"),
            "object": "chat.completion",
            "created": _now(),
            "model": request.model or app.state.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": choice_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": _build_usage(prompt_token_count, completion_token_count),
        }
        return JSONResponse(response)

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest) -> JSONResponse:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming is not implemented")

        engine = app.state.engine
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        outputs = await _run_generate(
            engine,
            prompts,
            SamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                ignore_eos=request.ignore_eos,
            ),
        )

        prompt_token_count = 0
        if prompts and isinstance(prompts[0], str):
            prompt_token_count = sum(len(engine.tokenizer.encode(prompt)) for prompt in prompts)
        elif prompts and isinstance(prompts[0], list):
            prompt_token_count = sum(len(prompt) for prompt in prompts)

        choices = []
        completion_token_count = 0
        for index, output in enumerate(outputs):
            completion_token_count += len(output["token_ids"])
            choices.append(
                {
                    "index": index,
                    "text": output["text"],
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            )

        response = {
            "id": _make_id("cmpl"),
            "object": "text_completion",
            "created": _now(),
            "model": request.model or app.state.model_name,
            "choices": choices,
            "usage": _build_usage(prompt_token_count, completion_token_count),
        }
        return JSONResponse(response)

    @app.post("/v1/responses")
    async def responses(request: ResponseCreateRequest):
        try:
            engine = app.state.engine
            messages = _messages_from_responses_input(request.input)
            prompt = _build_chat_prompt(engine, messages)
            prompt += _tool_prompt_suffix(request.tools, request.tool_choice)
            outputs = await _run_generate(
                engine,
                [prompt],
                SamplingParams(
                    temperature=request.temperature,
                    max_tokens=request.max_output_tokens,
                    ignore_eos=request.ignore_eos,
                ),
            )

            output_text = outputs[0]["text"]
            prompt_tokens = len(engine.tokenizer.encode(prompt))
            completion_tokens = len(outputs[0]["token_ids"])
            response_id = _make_id("resp")

            payload: dict[str, Any]
            is_function_call = False
            allowed_tools = _get_allowed_tool_names(request.tools)
            parsed = _extract_json_object(output_text) if allowed_tools else None
            if parsed and parsed.get("type") == "tool_call":
                tool_name = str(parsed.get("name", ""))
                arguments = parsed.get("arguments", {})
                if tool_name in allowed_tools and isinstance(arguments, dict):
                    arguments_json = json.dumps(arguments, ensure_ascii=True)
                    payload = _build_function_call_response_object(
                        response_id=response_id,
                        model=request.model or app.state.model_name,
                        tool_name=tool_name,
                        arguments_json=arguments_json,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    is_function_call = True
                else:
                    payload = _build_responses_object(
                        response_id=response_id,
                        model=request.model or app.state.model_name,
                        output_text=output_text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
            elif parsed and parsed.get("type") == "message" and isinstance(parsed.get("content"), str):
                payload = _build_responses_object(
                    response_id=response_id,
                    model=request.model or app.state.model_name,
                    output_text=parsed["content"],
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            else:
                payload = _build_responses_object(
                    response_id=response_id,
                    model=request.model or app.state.model_name,
                    output_text=output_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            if not request.stream:
                return JSONResponse(payload)

            async def event_gen():
                yield _sse_event({"type": "response.created", "response": payload})

                item = payload["output"][0]
                item_id = item["id"]

                if is_function_call:
                    yield _sse_event(
                        {
                            "type": "response.output_item.added",
                            "response_id": response_id,
                            "output_index": 0,
                            "item": {
                                "id": item_id,
                                "type": "function_call",
                                "status": "in_progress",
                                "call_id": item["call_id"],
                                "name": item["name"],
                                "arguments": "",
                            },
                        }
                    )
                    yield _sse_event(
                        {
                            "type": "response.function_call_arguments.delta",
                            "response_id": response_id,
                            "output_index": 0,
                            "item_id": item_id,
                            "delta": item["arguments"],
                        }
                    )
                    yield _sse_event(
                        {
                            "type": "response.function_call_arguments.done",
                            "response_id": response_id,
                            "output_index": 0,
                            "item_id": item_id,
                            "arguments": item["arguments"],
                        }
                    )
                    yield _sse_event(
                        {
                            "type": "response.output_item.done",
                            "response_id": response_id,
                            "output_index": 0,
                            "item": item,
                        }
                    )
                else:
                    content = item["content"][0]
                    text = content["text"]
                    yield _sse_event(
                        {
                            "type": "response.output_item.added",
                            "response_id": response_id,
                            "output_index": 0,
                            "item": {
                                "id": item_id,
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            },
                        }
                    )
                    yield _sse_event(
                        {
                            "type": "response.content_part.added",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        }
                    )
                    yield _sse_event(
                        {
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": text,
                        }
                    )
                    yield _sse_event(
                        {
                            "type": "response.output_text.done",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "text": text,
                        }
                    )
                    yield _sse_event(
                        {
                            "type": "response.content_part.done",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": content,
                        }
                    )
                    yield _sse_event(
                        {
                            "type": "response.output_item.done",
                            "response_id": response_id,
                            "output_index": 0,
                            "item": item,
                        }
                    )

                yield _sse_event({"type": "response.completed", "response": payload})
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_gen(), media_type="text/event-stream")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid responses request: {exc}") from exc

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": app.title,
            "model": app.state.model_name,
            "endpoints": ["/healthz", "/v1/models", "/v1/chat/completions", "/v1/completions", "/v1/responses"],
        }

    return app


app = create_app()


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Run the OpenAI-compatible API server")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", default=None)
    parser.add_argument("--ssl-certfile", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import uvicorn

    uvicorn.run(
        create_app(args.model_path),
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        reload=False,
    )


if __name__ == "__main__":
    main()