from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.config.sampling_params import SamplingParams
from src.engine.llm_engine import LLMEngine
from src.models.qwen3 import Qwen3ForCausalLM
from src.utils.html_templates import _index_html


DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Conversation roles follow OpenAI Chat format: system sets rules, user is the human, assistant is you. "
    "When asked about role or identity, answer directly and briefly based on message roles."
)


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[Any]
    name: str | None = None


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = Field(default=1, gt=1e-10)
    max_tokens: int = Field(default=512, ge=1)
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
    chat_messages = [
        {
            "role": message.role,
            "content": _normalize_text_content(message.content),
        }
        for message in messages
    ]

    # Ensure there is always one high-priority instruction that clarifies role semantics.
    if not any(item["role"] == "system" for item in chat_messages):
        chat_messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    return engine.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)


def _build_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _postprocess_qwen3_output(text: str) -> tuple[str, list[str]]:
  normalized = "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()
  return normalized, ["<think>", "</think>"]


def _extract_think_sections(text: str, think_tags: list[str]) -> tuple[str, list[str]]:
  if len(think_tags) != 2:
    return text.strip(), []

  start_tag, end_tag = think_tags
  pattern = re.compile(rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}", re.DOTALL)
  think_sections = [chunk.strip() for chunk in pattern.findall(text) if chunk.strip()]
  cleaned_text = pattern.sub("", text)
  cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
  return cleaned_text, think_sections


def _postprocess_model_output(
  text: str,
  model_name: str,
) -> tuple[str, list[str], list[str]]:
  if "qwen3" in model_name.lower():
    normalized, think_tags = _postprocess_qwen3_output(text)
  else:
    normalized, think_tags = text.strip(), []

  answer, think_sections = _extract_think_sections(normalized, think_tags)
  return answer, think_sections, think_tags

async def _run_generate(
    engine: LLMEngine,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
):
    return await asyncio.to_thread(engine.generate, prompts, sampling_params, False)


@lru_cache(maxsize=1)
def get_engine(model_path: str = DEFAULT_MODEL_PATH) -> LLMEngine:
    return LLMEngine(os.path.expanduser(model_path), Qwen3ForCausalLM)


def create_app(model_path: str = DEFAULT_MODEL_PATH) -> FastAPI:
    app = FastAPI(title="nano-vllm-diy Web Chat", version="0.1.0")

    @app.on_event("startup")
    def _warmup() -> None:
        app.state.engine = get_engine(model_path)
        app.state.model_name = os.path.expanduser(model_path).rstrip("/").split("/")[-1] or "model"

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "model": app.state.model_name,
            "title": app.title,
        }

    @app.get("/")
    def root() -> HTMLResponse:
        return HTMLResponse(_index_html())

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> JSONResponse:
        try:
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

            answer, think_sections, think_tags = _postprocess_model_output(
              outputs[0]["text"],
              app.state.model_name,
            )
            return JSONResponse(
                {
                    "id": _make_id("chatcmpl"),
                    "object": "chat.message",
                    "created": _now(),
                    "model": app.state.model_name,
                    "answer": answer,
                "think": think_sections,
                "think_tags": think_tags,
                    "usage": _build_usage(
                        len(engine.tokenizer.encode(prompt)),
                        len(outputs[0]["token_ids"]),
                    ),
                }
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid chat request: {exc}") from exc

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest) -> JSONResponse:
        try:
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

            answer, think_sections, think_tags = _postprocess_model_output(
              outputs[0]["text"],
              app.state.model_name,
            )
            response = {
                "id": _make_id("chatcmpl"),
                "object": "chat.completion",
                "created": _now(),
                "model": app.state.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _build_usage(
                    len(engine.tokenizer.encode(prompt)),
                    len(outputs[0]["token_ids"]),
                ),
                "think": think_sections,
                "think_tags": think_tags,
            }
            return JSONResponse(response)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid chat request: {exc}") from exc

    return app


app = create_app()


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Run the web chat server")
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