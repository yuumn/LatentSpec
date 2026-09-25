#!/usr/bin/env python3
"""Benchmark aggregate TPS against an OpenAI-compatible SGLang server."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import aiohttp


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR.parent / "datasets"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "tps_results"
WARMUP_CONCURRENCY = 16


@dataclass
class RequestResult:
    success: bool
    request_start_time: float
    finish_time: float
    completion_tokens: int = 0
    first_token_time: float | None = None
    last_token_time: float | None = None
    error: str | None = None

    @property
    def ttft_seconds(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.request_start_time

    @property
    def decode_tokens(self) -> int:
        return max(self.completion_tokens - 1, 0)

    @property
    def decode_seconds(self) -> float | None:
        if self.first_token_time is None or self.last_token_time is None:
            return None
        return max(self.last_token_time - self.first_token_time, 0.0)

    @property
    def decode_tps(self) -> float | None:
        decode_seconds = self.decode_seconds
        if self.decode_tokens <= 0 or decode_seconds is None or decode_seconds <= 0:
            return None
        return self.decode_tokens / decode_seconds

    @property
    def tpot_seconds(self) -> float | None:
        decode_seconds = self.decode_seconds
        if self.decode_tokens <= 0 or decode_seconds is None or decode_seconds <= 0:
            return None
        return decode_seconds / self.decode_tokens


def parse_name_list(values: list[str]) -> list[str]:
    result = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def parse_concurrencies(values: list[str]) -> list[int]:
    parsed = []
    for value in parse_name_list(values):
        concurrency = int(value)
        if concurrency <= 0:
            raise argparse.ArgumentTypeError("concurrencies must be positive integers")
        if concurrency not in parsed:
            parsed.append(concurrency)
    if not parsed:
        raise argparse.ArgumentTypeError("at least one concurrency is required")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure aggregate TPS with streaming OpenAI chat requests, both "
            "including and excluding the first generated token and its latency."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--model",
        default=None,
        help="Served model name. By default it is discovered from /v1/models.",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset stems separated by spaces or commas; use 'all' for every JSONL file.",
    )
    parser.add_argument(
        "--concurrencies",
        nargs="+",
        default=["1", "2", "4", "8", "16"],
        help="Maximum concurrent request counts, separated by spaces or commas.",
    )
    parser.add_argument(
        "--requests-per-dataset",
        type=int,
        default=0,
        help="Prompts used per dataset and concurrency; 0 means all prompts.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=16,
        help="Number of warmup requests. They are sent with concurrency 16.",
    )

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--max-tokens", type=int, default=256)
    sampling.add_argument("--temperature", type=float, default=1.0)
    sampling.add_argument("--top-p", type=float, default=1.0)
    sampling.add_argument("--top-k", type=int, default=-1)
    sampling.add_argument("--min-p", type=float, default=0.0)
    sampling.add_argument("--frequency-penalty", type=float, default=0.0)
    sampling.add_argument("--presence-penalty", type=float, default=0.0)
    sampling.add_argument("--repetition-penalty", type=float, default=1.0)
    sampling.add_argument("--sampling-seed", type=int, default=None)
    sampling.add_argument("--stop", action="append", default=None)
    sampling.add_argument("--ignore-eos", action="store_true")
    sampling.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model thinking in the chat template; disabled by default.",
    )
    sampling.add_argument(
        "--extra-request-body",
        default=None,
        help="A JSON object merged into each OpenAI request body.",
    )

    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--flush-timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Optional JSONL filename. A timestamped name is used by default.",
    )
    args = parser.parse_args()

    try:
        args.concurrencies = parse_concurrencies(args.concurrencies)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    args.datasets = parse_name_list(args.datasets)
    if not args.datasets:
        parser.error("at least one dataset is required")
    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.requests_per_dataset < 0:
        parser.error("--requests-per-dataset must be non-negative")
    if args.warmup_requests < WARMUP_CONCURRENCY:
        parser.error(
            f"--warmup-requests must be at least {WARMUP_CONCURRENCY} so the "
            f"required concurrency-{WARMUP_CONCURRENCY} warmup can be reached"
        )
    if args.max_tokens <= 1:
        parser.error("--max-tokens must be greater than 1 for decode TPS")
    if args.request_timeout <= 0 or args.flush_timeout <= 0:
        parser.error("timeouts must be positive")

    if args.extra_request_body is not None:
        try:
            args.extra_request_body = json.loads(args.extra_request_body)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid --extra-request-body JSON: {exc}")
        if not isinstance(args.extra_request_body, dict):
            parser.error("--extra-request-body must decode to a JSON object")
    else:
        args.extra_request_body = {}
    return args


def load_dataset(path: Path) -> list[str]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    f"{path}:{line_number}: field 'prompt' must be a non-empty string"
                )
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"dataset is empty: {path}")
    return prompts


def resolve_datasets(
    dataset_dir: Path,
    requested_names: list[str],
    requests_per_dataset: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")

    available_paths = {path.stem: path for path in sorted(dataset_dir.glob("*.jsonl"))}
    if not available_paths:
        raise FileNotFoundError(f"no JSONL datasets found in: {dataset_dir}")
    if "all" in requested_names:
        if len(requested_names) != 1:
            raise ValueError("'all' cannot be combined with explicit dataset names")
        dataset_names = list(available_paths)
    else:
        dataset_names = requested_names

    missing = [name for name in dataset_names if name not in available_paths]
    if missing:
        available = ", ".join(available_paths)
        raise ValueError(
            f"unknown datasets: {', '.join(missing)}; available datasets: {available}"
        )

    datasets = {}
    dataset_metadata = {}
    for name in dataset_names:
        path = available_paths[name]
        all_prompts = load_dataset(path)
        selected_prompts = (
            all_prompts[:requests_per_dataset]
            if requests_per_dataset > 0
            else all_prompts
        )
        datasets[name] = selected_prompts
        dataset_metadata[name] = {
            "path": str(path),
            "available_prompts": len(all_prompts),
            "selected_prompts": len(selected_prompts),
        }
    return datasets, dataset_metadata


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def discover_model(
    session: aiohttp.ClientSession,
    base_url: str,
    headers: dict[str, str],
) -> str:
    url = f"{base_url}/v1/models"
    async with session.get(url, headers=headers) as response:
        response_text = await response.text()
        if response.status != 200:
            raise RuntimeError(
                f"failed to discover model from {url}: HTTP {response.status}: "
                f"{response_text[:500]}"
            )
        payload = json.loads(response_text)
    models = payload.get("data") or []
    if not models or not isinstance(models[0].get("id"), str):
        raise RuntimeError(f"no model id returned by {url}; pass --model explicitly")
    return models[0]["id"]


def build_request_body(args: argparse.Namespace, model: str, prompt: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "ignore_eos": args.ignore_eos,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }
    if args.sampling_seed is not None:
        body["seed"] = args.sampling_seed
    if args.stop:
        body["stop"] = args.stop
    body.update(args.extra_request_body)

    # Streaming and usage are required for decode-only timing and token counts.
    body["stream"] = True
    stream_options = body.get("stream_options")
    if not isinstance(stream_options, dict):
        stream_options = {}
    body["stream_options"] = {**stream_options, "include_usage": True}
    return body


def streamed_text(delta: dict[str, Any]) -> str:
    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
    content = delta.get("content") or ""
    return f"{reasoning}{content}"


async def send_request(
    *,
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> RequestResult:
    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    completion_tokens = None
    try:
        async with session.post(url, headers=headers, json=body) as response:
            if response.status != 200:
                response_text = await response.text()
                raise RuntimeError(
                    f"HTTP {response.status}: {response_text[:1000]}"
                )

            while not response.content.at_eof():
                line = await response.content.readline()
                if not line:
                    break
                line_text = line.decode("utf-8").strip()
                if not line_text or line_text.startswith(":"):
                    continue
                if not line_text.startswith("data:"):
                    continue
                event_text = line_text[5:].strip()
                if event_text == "[DONE]":
                    break

                event = json.loads(event_text)
                usage = event.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])

                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if streamed_text(delta):
                    token_time = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = token_time
                    last_token_time = token_time

        finish_time = time.perf_counter()
        if completion_tokens is None:
            raise RuntimeError(
                "stream ended without usage.completion_tokens; the server must "
                "support stream_options.include_usage"
            )
        if completion_tokens > 0 and first_token_time is None:
            raise RuntimeError("completion tokens were reported but no token event was received")
        return RequestResult(
            success=True,
            request_start_time=start_time,
            finish_time=finish_time,
            completion_tokens=completion_tokens,
            first_token_time=first_token_time,
            last_token_time=last_token_time,
        )
    except Exception as exc:
        return RequestResult(
            success=False,
            request_start_time=start_time,
            finish_time=time.perf_counter(),
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_requests(
    *,
    session: aiohttp.ClientSession,
    url: str,
    headers: dict[str, str],
    model: str,
    prompts: list[str],
    concurrency: int,
    args: argparse.Namespace,
) -> tuple[list[RequestResult], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(prompt: str) -> RequestResult:
        async with semaphore:
            return await send_request(
                session=session,
                url=url,
                headers=headers,
                body=build_request_body(args, model, prompt),
            )

    start_time = time.perf_counter()
    results = await asyncio.gather(
        *(asyncio.create_task(limited_request(prompt)) for prompt in prompts)
    )
    return results, time.perf_counter() - start_time


async def flush_cache(
    session: aiohttp.ClientSession,
    base_url: str,
    headers: dict[str, str],
    timeout_seconds: float,
) -> None:
    url = f"{base_url}/flush_cache"
    last_error = ""
    for attempt in range(1, 4):
        try:
            async with session.post(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as response:
                response_text = await response.text()
                if response.status == 200:
                    print(f"Flushed server cache: {url}", flush=True)
                    return
                last_error = f"HTTP {response.status}: {response_text[:500]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < 3:
            await asyncio.sleep(1.0)
    raise RuntimeError(f"failed to flush cache at {url}: {last_error}")


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_distribution_ms(values_seconds: list[float]) -> dict[str, float | None]:
    values_ms = [value * 1000.0 for value in values_seconds]
    return {
        "avg": statistics.fmean(values_ms) if values_ms else None,
        "p50": percentile(values_ms, 0.50),
        "p95": percentile(values_ms, 0.95),
        "p99": percentile(values_ms, 0.99),
    }


def interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Return wall time during which at least one request was decoding."""
    positive_intervals = sorted(
        (start, end) for start, end in intervals if end > start
    )
    if not positive_intervals:
        return 0.0

    total = 0.0
    current_start, current_end = positive_intervals[0]
    for start, end in positive_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def summarize_group(
    *,
    dataset_name: str,
    concurrency: int,
    results: list[RequestResult],
    wall_time_seconds: float,
) -> dict[str, Any]:
    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]
    timed = [
        result
        for result in successful
        if result.first_token_time is not None and result.last_token_time is not None
    ]
    total_completion_tokens = sum(result.completion_tokens for result in successful)
    total_decode_tokens = sum(result.decode_tokens for result in successful)

    time_including_first_token_seconds = interval_union_seconds(
        [
            (result.request_start_time, result.last_token_time)
            for result in timed
            if result.last_token_time is not None
        ]
    )
    time_excluding_first_token_seconds = interval_union_seconds(
        [
            (result.first_token_time, result.last_token_time)
            for result in timed
            if result.first_token_time is not None and result.last_token_time is not None
        ]
    )
    tps_including_first_token = (
        total_completion_tokens / time_including_first_token_seconds
        if total_completion_tokens > 0 and time_including_first_token_seconds > 0
        else None
    )
    tps_excluding_first_token = (
        total_decode_tokens / time_excluding_first_token_seconds
        if total_decode_tokens > 0 and time_excluding_first_token_seconds > 0
        else None
    )

    ttfts = [
        result.ttft_seconds
        for result in successful
        if result.ttft_seconds is not None
    ]
    tpots = [
        result.tpot_seconds
        for result in successful
        if result.tpot_seconds is not None
    ]
    request_decode_tps = [
        result.decode_tps for result in successful if result.decode_tps is not None
    ]
    return {
        "dataset": dataset_name,
        "concurrency": concurrency,
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "wall_time_seconds": wall_time_seconds,
        "total_completion_tokens": total_completion_tokens,
        "decode_tokens_excluding_first": total_decode_tokens,
        "time_including_first_token_seconds": time_including_first_token_seconds,
        "time_excluding_first_token_seconds": time_excluding_first_token_seconds,
        "tps_including_first_token": tps_including_first_token,
        "tps_excluding_first_token": tps_excluding_first_token,
        "request_throughput_requests_per_second": (
            len(successful) / wall_time_seconds if wall_time_seconds > 0 else None
        ),
        "ttft_ms": latency_distribution_ms(ttfts),
        "tpot_ms": latency_distribution_ms(tpots),
        "per_request_decode_tps": {
            "mean": statistics.fmean(request_decode_tps)
            if request_decode_tps
            else None,
            "median": statistics.median(request_decode_tps)
            if request_decode_tps
            else None,
            "p90": percentile(request_decode_tps, 0.90),
        },
        "sample_errors": [result.error for result in failed[:5]],
    }


def format_tps(value: float | None, failed_requests: int) -> str:
    if value is None:
        text = "N/A"
    else:
        text = f"{value:.2f}"
    return f"{text}*" if failed_requests else text


def print_tps_table(
    dataset_names: list[str],
    concurrencies: list[int],
    results: list[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
) -> None:
    result_lookup = {
        (result["dataset"], result["concurrency"]): result for result in results
    }
    headers = ["dataset", *(f"c={concurrency}" for concurrency in concurrencies)]
    rows = []
    for dataset_name in dataset_names:
        row = [dataset_name]
        for concurrency in concurrencies:
            result = result_lookup.get((dataset_name, concurrency))
            if result is None:
                row.append("-")
            else:
                row.append(
                    format_tps(
                        result[metric_key],
                        result["failed_requests"],
                    )
                )
        rows.append(row)

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    print(f"\n{title}")
    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))
    if any(result["failed_requests"] for result in results):
        print("* One or more requests failed; inspect the JSON result for details.")


def format_latency_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def print_latency_table(results: list[dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "concurrency",
        "TTFT avg",
        "TTFT p50",
        "TTFT p95",
        "TTFT p99",
        "TPOT avg",
        "TPOT p50",
        "TPOT p95",
        "TPOT p99",
    ]
    rows = []
    for result in results:
        ttft = result["ttft_ms"]
        tpot = result["tpot_ms"]
        rows.append(
            [
                result["dataset"],
                str(result["concurrency"]),
                *(format_latency_ms(ttft[key]) for key in ("avg", "p50", "p95", "p99")),
                *(format_latency_ms(tpot[key]) for key in ("avg", "p50", "p95", "p99")),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    print("\nLatency distribution (ms)")
    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def sampling_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_completion_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.sampling_seed,
        "stop": args.stop,
        "ignore_eos": args.ignore_eos,
        "enable_thinking": args.enable_thinking,
        "extra_request_body": args.extra_request_body,
    }


def metric_definitions() -> dict[str, str]:
    return {
        "tps_including_first_token": (
            "sum(completion_tokens) / union_duration_of_each_successful_request_"
            "interval_from_request_start_to_last_generated_token"
        ),
        "tps_excluding_first_token": (
            "sum(max(completion_tokens - 1, 0)) / union_duration_of_each_"
            "successful_request_interval_from_first_generated_token_to_last_"
            "generated_token"
        ),
    }


def build_tps_table(
    *,
    dataset_names: list[str],
    concurrencies: list[int],
    results: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    return {
        metric_name: {
            dataset_name: {
                str(concurrency): next(
                    (
                        result[metric_name]
                        for result in results
                        if result["dataset"] == dataset_name
                        and result["concurrency"] == concurrency
                    ),
                    None,
                )
                for concurrency in concurrencies
            }
            for dataset_name in dataset_names
        }
        for metric_name in (
            "tps_including_first_token",
            "tps_excluding_first_token",
        )
    }


def build_latency_table(
    *,
    dataset_names: list[str],
    concurrencies: list[int],
    results: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    result_lookup = {
        (result["dataset"], result["concurrency"]): result for result in results
    }
    return {
        dataset_name: {
            str(concurrency): {
                "ttft_ms": result_lookup[(dataset_name, concurrency)]["ttft_ms"],
                "tpot_ms": result_lookup[(dataset_name, concurrency)]["tpot_ms"],
            }
            for concurrency in concurrencies
            if (dataset_name, concurrency) in result_lookup
        }
        for dataset_name in dataset_names
    }


def prepare_output_file(output_dir: Path, filename: str | None) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"decode_tps_{timestamp}.jsonl"
    elif not Path(filename).suffix:
        filename = f"{filename}.jsonl"
    output_path = output_dir / filename
    output_path.write_text("", encoding="utf-8")
    return output_path


def append_jsonl_record(output_path: Path, record: dict[str, Any]) -> None:
    """Durably append one complete JSON object to the result file."""
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


async def benchmark(args: argparse.Namespace) -> Path:
    datasets, dataset_metadata = resolve_datasets(
        args.dataset_dir,
        args.datasets,
        args.requests_per_dataset,
    )
    dataset_names = list(datasets)
    all_selected_prompts = [prompt for prompts in datasets.values() for prompt in prompts]
    if not all_selected_prompts:
        raise RuntimeError("no prompts were selected")

    base_url = f"http://{args.host}:{args.port}"
    chat_url = f"{base_url}/v1/chat/completions"
    headers = request_headers(args.api_key)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        model = args.model or await discover_model(session, base_url, headers)
        print(f"Server: {base_url}")
        print(f"Model: {model}")
        print(f"Datasets: {', '.join(dataset_names)}")

        output_path = prepare_output_file(args.output_dir, args.output_filename)
        append_jsonl_record(
            output_path,
            {
                "record_type": "metadata",
                "created_at": datetime.now().astimezone().isoformat(),
                "server": {
                    "base_url": base_url,
                    "port": args.port,
                    "model": model,
                },
                "benchmark": {
                    "concurrencies": args.concurrencies,
                    "requests_per_dataset": args.requests_per_dataset,
                    "warmup_requests": args.warmup_requests,
                    "warmup_concurrency": WARMUP_CONCURRENCY,
                    "sampling": sampling_config(args),
                    "metric_definitions": metric_definitions(),
                    "latency_definitions": {
                        "ttft_ms": "request_start_to_first_generated_token",
                        "tpot_ms": (
                            "(last_generated_token_time - first_generated_token_time) "
                            "/ max(completion_tokens - 1, 0)"
                        ),
                    },
                },
                "datasets": dataset_metadata,
            },
        )
        print(f"Incremental JSONL output: {output_path}", flush=True)

        warmup_prompts = list(islice(cycle(all_selected_prompts), args.warmup_requests))
        print(
            f"Starting {args.warmup_requests} warmup requests with "
            f"concurrency {WARMUP_CONCURRENCY}...",
            flush=True,
        )
        warmup_results, warmup_seconds = await run_requests(
            session=session,
            url=chat_url,
            headers=headers,
            model=model,
            prompts=warmup_prompts,
            concurrency=WARMUP_CONCURRENCY,
            args=args,
        )
        warmup_failures = [result for result in warmup_results if not result.success]
        print(
            f"Warmup completed in {warmup_seconds:.2f}s: "
            f"{len(warmup_results) - len(warmup_failures)}/{len(warmup_results)} succeeded.",
            flush=True,
        )
        await flush_cache(session, base_url, headers, args.flush_timeout)
        if warmup_failures:
            raise RuntimeError(
                f"{len(warmup_failures)} warmup requests failed; first error: "
                f"{warmup_failures[0].error}"
            )

        benchmark_results = []
        for concurrency in args.concurrencies:
            for dataset_name, prompts in datasets.items():
                print(
                    f"Testing dataset={dataset_name}, concurrency={concurrency}, "
                    f"requests={len(prompts)}...",
                    flush=True,
                )
                try:
                    request_results, wall_time_seconds = await run_requests(
                        session=session,
                        url=chat_url,
                        headers=headers,
                        model=model,
                        prompts=prompts,
                        concurrency=concurrency,
                        args=args,
                    )
                    summary = summarize_group(
                        dataset_name=dataset_name,
                        concurrency=concurrency,
                        results=request_results,
                        wall_time_seconds=wall_time_seconds,
                    )
                    benchmark_results.append(summary)
                    append_jsonl_record(
                        output_path,
                        {
                            "record_type": "result",
                            "created_at": datetime.now().astimezone().isoformat(),
                            **summary,
                        },
                    )
                    tps_including_text = format_tps(
                        summary["tps_including_first_token"],
                        summary["failed_requests"],
                    )
                    tps_excluding_text = format_tps(
                        summary["tps_excluding_first_token"],
                        summary["failed_requests"],
                    )
                    print(
                        f"Completed dataset={dataset_name}, concurrency={concurrency}: "
                        f"including_first_token(time="
                        f"{summary['time_including_first_token_seconds']:.4f}s, "
                        f"tps={tps_including_text}), excluding_first_token(time="
                        f"{summary['time_excluding_first_token_seconds']:.4f}s, "
                        f"tps={tps_excluding_text}), successful="
                        f"{summary['successful_requests']}/{summary['total_requests']}",
                        flush=True,
                    )
                    print(
                        "  TTFT(ms): "
                        + ", ".join(
                            f"{key}={format_latency_ms(summary['ttft_ms'][key])}"
                            for key in ("avg", "p50", "p95", "p99")
                        ),
                        flush=True,
                    )
                    print(
                        "  TPOT(ms): "
                        + ", ".join(
                            f"{key}={format_latency_ms(summary['tpot_ms'][key])}"
                            for key in ("avg", "p50", "p95", "p99")
                        ),
                        flush=True,
                    )
                    print(
                        f"Incrementally saved dataset={dataset_name}, "
                        f"concurrency={concurrency} to {output_path}",
                        flush=True,
                    )
                finally:
                    await flush_cache(session, base_url, headers, args.flush_timeout)

    print_tps_table(
        dataset_names,
        args.concurrencies,
        benchmark_results,
        metric_key="tps_including_first_token",
        title="Aggregate TPS including first-token latency (tokens/s)",
    )
    print_tps_table(
        dataset_names,
        args.concurrencies,
        benchmark_results,
        metric_key="tps_excluding_first_token",
        title="Aggregate decode TPS excluding first-token latency (tokens/s)",
    )
    print_latency_table(benchmark_results)
    append_jsonl_record(
        output_path,
        {
            "record_type": "summary",
            "created_at": datetime.now().astimezone().isoformat(),
            "completed_result_count": len(benchmark_results),
            "expected_result_count": len(dataset_names) * len(args.concurrencies),
            "tps_table": build_tps_table(
                dataset_names=dataset_names,
                concurrencies=args.concurrencies,
                results=benchmark_results,
            ),
            "latency_table": build_latency_table(
                dataset_names=dataset_names,
                concurrencies=args.concurrencies,
                results=benchmark_results,
            ),
        },
    )
    print(f"\nSaved incremental JSONL results to: {output_path}", flush=True)
    return output_path


def main() -> None:
    args = parse_args()
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
