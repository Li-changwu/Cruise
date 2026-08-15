"""OpenAI/vLLM-compatible HTTP entry for the Persistent Device Model Owner."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, AsyncIterator, Callable
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import tokenizers
from transformers import PreTrainedTokenizerFast

from .owner_transport import OwnerEvent, OwnerRequest, OwnerTransport


@dataclass(frozen=True)
class ServiceSettings:
    served_model_name: str
    owner_command: tuple[str, ...]
    batch_wait_ms: float = 5.0
    admission_cohort_size: int | None = None
    eos_token_id: int = 151645
    owner_startup_timeout: float = 1140.0


class IncrementalTokenDecoder:
    """Per-request vLLM-compatible fast detokenization state."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, prompt_tokens: tuple[int, ...]
    ) -> None:
        self._backend = tokenizer._tokenizer
        self._stream = tokenizers.decoders.DecodeStream(
            ids=list(prompt_tokens), skip_special_tokens=True
        )
        self.tokens: list[int] = []
        self.text = ""

    def push(self, token: int) -> str:
        delta = self._stream.step(self._backend, token) or ""
        self.tokens.append(token)
        self.text += delta
        return delta


class TokenCodec:
    def __init__(self, tokenizer: str | Path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer), local_files_only=True, trust_remote_code=False
        )
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            raise TypeError("P5 requires a fast tokenizer for incremental decoding")
        if not hasattr(tokenizers.decoders, "DecodeStream"):
            raise RuntimeError("P5 requires tokenizers.decoders.DecodeStream")

    def incremental(
        self, prompt_tokens: tuple[int, ...]
    ) -> IncrementalTokenDecoder:
        return IncrementalTokenDecoder(self.tokenizer, prompt_tokens)


class AdmissionBatcher:
    def __init__(
        self,
        transport: OwnerTransport,
        wait_ms: float,
        cohort_size: int | None = None,
    ) -> None:
        if cohort_size is not None and not 1 <= cohort_size <= 4:
            raise ValueError("admission cohort size must be in [1, 4]")
        self.transport = transport
        self.wait_seconds = wait_ms / 1000.0
        self.cohort_size = cohort_size
        self.pending: asyncio.Queue[tuple[OwnerRequest, bool]] = asyncio.Queue()
        self.next_request = 1
        self.next_generation = 1
        self.next_cohort = 1
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def submit(
        self, prompt: tuple[int, ...], max_tokens: int, ignore_eos: bool
    ) -> OwnerRequest:
        request = OwnerRequest(
            request=self.next_request,
            generation=self.next_generation,
            prompt_tokens=prompt,
            max_tokens=max_tokens,
        )
        self.next_request += 1
        self.next_generation += 1
        await self.pending.put((request, ignore_eos))
        return request

    async def _run(self) -> None:
        while True:
            batch: list[tuple[OwnerRequest, bool]] = []
            try:
                first = await self.pending.get()
                batch = [first]
                ignore_eos = first[1]
                if self.cohort_size is not None:
                    while len(batch) < self.cohort_size:
                        candidate = await self.pending.get()
                        if candidate[1] != ignore_eos:
                            candidate[0].events.put_nowait(
                                RuntimeError(
                                    "strict admission cohort settings differ"
                                )
                            )
                            continue
                        batch.append(candidate)
                else:
                    deadline = asyncio.get_running_loop().time() + self.wait_seconds
                    while len(batch) < 4:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            break
                        try:
                            candidate = await asyncio.wait_for(
                                self.pending.get(), timeout=remaining
                            )
                        except asyncio.TimeoutError:
                            break
                        if candidate[1] != ignore_eos:
                            await self.pending.put(candidate)
                            break
                        batch.append(candidate)
                requests = [item[0] for item in batch]
                await self.transport.admit_many(
                    requests,
                    cohort_id=self.next_cohort,
                    ignore_eos=ignore_eos,
                    eos_token=151645,
                )
            except asyncio.CancelledError:
                for request, _ in batch:
                    request.events.put_nowait(RuntimeError("admission batcher closed"))
                raise
            except Exception as exc:
                for request, _ in batch:
                    request.events.put_nowait(exc)
            self.next_cohort += 1

    async def close(self) -> None:
        if self.task is None:
            return
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        while not self.pending.empty():
            request, _ = self.pending.get_nowait()
            request.events.put_nowait(RuntimeError("admission batcher closed"))
        self.task = None


def _validate_request(body: Any, settings: ServiceSettings) -> tuple[tuple[int, ...], int, bool, bool, bool]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")
    if body.get("model") != settings.served_model_name:
        raise HTTPException(status_code=404, detail="unknown model")
    prompt = body.get("prompt")
    if (
        not isinstance(prompt, list)
        or not 1 <= len(prompt) <= 128
        or not all(isinstance(token, int) and 0 <= token < 152064 for token in prompt)
    ):
        raise HTTPException(status_code=400, detail="prompt must be 1-128 token IDs")
    max_tokens = body.get("max_tokens", 16)
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 256:
        raise HTTPException(status_code=400, detail="max_tokens must be in [1, 256]")
    if len(prompt) + max_tokens > 384:
        raise HTTPException(status_code=400, detail="request exceeds Device KV lease")
    supported = {
        "temperature": (0, 0.0, None),
        "top_p": (1, 1.0, None),
        "top_k": (0, -1, None),
        "min_p": (0, 0.0, None),
        "presence_penalty": (0, 0.0, None),
        "frequency_penalty": (0, 0.0, None),
        "repetition_penalty": (1, 1.0, None),
        "min_tokens": (0, None),
        "n": (1, None),
    }
    for name, values in supported.items():
        if body.get(name) not in values:
            raise HTTPException(
                status_code=400, detail=f"Persistent Owner does not support {name}"
            )
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise HTTPException(status_code=400, detail="stream must be boolean")
    ignore_eos = body.get("ignore_eos", False)
    if not isinstance(ignore_eos, bool):
        raise HTTPException(status_code=400, detail="ignore_eos must be boolean")
    return_token_ids = body.get("return_token_ids", False)
    if not isinstance(return_token_ids, bool):
        raise HTTPException(status_code=400, detail="return_token_ids must be boolean")
    return tuple(prompt), max_tokens, stream, ignore_eos, return_token_ids


def _finish_reason(event: OwnerEvent) -> str:
    if event.finish_reason == 1:
        return "stop"
    if event.finish_reason == 2:
        return "length"
    return "cancelled"


def _choice(
    *, text: str, token_ids: list[int], finish_reason: str | None, include_ids: bool
) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "index": 0,
        "text": text,
        "logprobs": None,
        "finish_reason": finish_reason,
        "stop_reason": None,
    }
    if include_ids:
        choice["token_ids"] = token_ids
    return choice


def _payload(
    *, completion_id: str, model: str, choices: list[dict[str, Any]], usage: dict[str, int] | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        result["usage"] = usage
    return result


async def _next_event(owner_request: OwnerRequest) -> OwnerEvent:
    value = await owner_request.events.get()
    if isinstance(value, BaseException):
        raise value
    if value.kind == "rejected":
        raise RuntimeError(f"Persistent Owner rejected request with status {value.status}")
    return value


async def _collect_completion(
    owner_request: OwnerRequest, codec: TokenCodec
) -> tuple[list[int], str, str]:
    decoder = codec.incremental(owner_request.prompt_tokens)
    while True:
        event = await _next_event(owner_request)
        if event.kind == "commit":
            decoder.push(event.token)
        elif event.kind.startswith("retire_"):
            return decoder.tokens, decoder.text, _finish_reason(event)


def create_app(
    settings: ServiceSettings,
    codec: TokenCodec,
    transport_factory: Callable[..., OwnerTransport] = OwnerTransport,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        transport = transport_factory(
            settings.owner_command,
            startup_timeout=settings.owner_startup_timeout,
        )
        await transport.start()
        batcher = AdmissionBatcher(
            transport,
            settings.batch_wait_ms,
            settings.admission_cohort_size,
        )
        batcher.start()
        app.state.owner_transport = transport
        app.state.admission_batcher = batcher
        try:
            yield
        finally:
            await batcher.close()
            await transport.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        metrics = app.state.owner_transport.metrics()
        if metrics["forbidden_runtime_modules"]:
            raise HTTPException(status_code=503, detail="forbidden legacy route loaded")
        return {"status": "ok", "route": metrics["route"]}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.served_model_name,
                    "object": "model",
                    "owned_by": "cruise-persistent-owner",
                }
            ],
        }

    @app.get("/cruise/metrics")
    async def metrics() -> dict[str, Any]:
        return app.state.owner_transport.metrics()

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        body = await request.json()
        prompt, max_tokens, stream, ignore_eos, include_ids = _validate_request(
            body, settings
        )
        owner_request = await app.state.admission_batcher.submit(
            prompt, max_tokens, ignore_eos
        )
        completion_id = f"cmpl-cruise-{uuid.uuid4().hex}"
        usage = {
            "prompt_tokens": len(prompt),
            "completion_tokens": max_tokens,
            "total_tokens": len(prompt) + max_tokens,
        }
        if not stream:
            try:
                tokens, text, finish = await _collect_completion(owner_request, codec)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            usage["completion_tokens"] = len(tokens)
            usage["total_tokens"] = len(prompt) + len(tokens)
            return JSONResponse(
                _payload(
                    completion_id=completion_id,
                    model=settings.served_model_name,
                    choices=[
                        _choice(
                            text=text,
                            token_ids=tokens,
                            finish_reason=finish,
                            include_ids=include_ids,
                        )
                    ],
                    usage=usage,
                )
            )

        async def generate() -> AsyncIterator[str]:
            decoder = codec.incremental(owner_request.prompt_tokens)
            retired = False
            try:
                while True:
                    event = await _next_event(owner_request)
                    if event.kind == "commit":
                        delta = decoder.push(event.token)
                        chunk = _payload(
                            completion_id=completion_id,
                            model=settings.served_model_name,
                            choices=[
                                _choice(
                                    text=delta,
                                    token_ids=[event.token],
                                    finish_reason=None,
                                    include_ids=include_ids,
                                )
                            ],
                        )
                        yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
                    elif event.kind.startswith("retire_"):
                        retired = True
                        finish = _finish_reason(event)
                        final = _payload(
                            completion_id=completion_id,
                            model=settings.served_model_name,
                            choices=[
                                _choice(
                                    text="",
                                    token_ids=[],
                                    finish_reason=finish,
                                    include_ids=include_ids,
                                )
                            ],
                        )
                        yield f"data: {json.dumps(final, separators=(',', ':'))}\n\n"
                        if body.get("stream_options", {}).get("include_usage"):
                            usage["completion_tokens"] = len(decoder.tokens)
                            usage["total_tokens"] = len(prompt) + len(decoder.tokens)
                            usage_chunk = _payload(
                                completion_id=completion_id,
                                model=settings.served_model_name,
                                choices=[],
                                usage=usage,
                            )
                            yield f"data: {json.dumps(usage_chunk, separators=(',', ':'))}\n\n"
                        yield "data: [DONE]\n\n"
                        return
            finally:
                if not retired:
                    await app.state.owner_transport.cancel(owner_request)

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--owner-executable", type=Path, required=True)
    parser.add_argument("--function-config", type=Path, required=True)
    parser.add_argument("--graph-config", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument("--air", type=Path, required=True)
    parser.add_argument("--owner-id", type=int, required=True)
    parser.add_argument("--batch-wait-ms", type=float, default=5.0)
    parser.add_argument(
        "--admission-cohort-size", type=int, choices=range(5), default=0
    )
    parser.add_argument("--owner-startup-timeout", type=float, default=1140.0)
    args = parser.parse_args()
    required_paths = (
        args.tokenizer,
        args.owner_executable,
        args.function_config,
        args.graph_config,
        args.deploy_config,
        args.air,
    )
    for path in required_paths:
        if not path.exists():
            parser.error(f"required path does not exist: {path}")
    settings = ServiceSettings(
        served_model_name=args.served_model_name,
        owner_command=(
            str(args.owner_executable),
            str(args.function_config),
            str(args.graph_config),
            str(args.deploy_config),
            str(args.air),
            str(args.owner_id),
        ),
        batch_wait_ms=args.batch_wait_ms,
        admission_cohort_size=args.admission_cohort_size or None,
        owner_startup_timeout=args.owner_startup_timeout,
    )
    import uvicorn

    uvicorn.run(
        create_app(settings, TokenCodec(args.tokenizer)),
        host=args.host,
        port=args.port,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
