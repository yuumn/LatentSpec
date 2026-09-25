#!/usr/bin/env python3
"""Compare the configured MySpec model with the configured DFlash baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from flops_core import (
    ModelDimensions,
    architecture_from_config,
    render_markdown_report,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MYSPEC_CONFIG = REPO_ROOT / "config/myspec/myspec_qwen3_4b_eager.py"
DEFAULT_DFLASH_CONFIG = REPO_ROOT.parent / "DeepSpec/config/dflash/dflash_qwen3_4b.py"


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate MySpec vs DFlash training/inference FLOPs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--myspec-config",
        type=Path,
        default=DEFAULT_MYSPEC_CONFIG,
        help="MySpec experiment config",
    )
    parser.add_argument(
        "--dflash-config",
        type=Path,
        default=DEFAULT_DFLASH_CONFIG,
        help="DFlash experiment config",
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        help="optional Qwen config.json; built-in defaults match Qwen3-4B",
    )
    parser.add_argument(
        "--context-lengths",
        type=positive_int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="training sequence lengths and inference committed contexts",
    )
    parser.add_argument(
        "--new-context-tokens",
        type=positive_int,
        help="steady-state uncached target states R; default is block_size+1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional Markdown output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dims = (
        ModelDimensions.from_json(args.target_config)
        if args.target_config is not None
        else ModelDimensions()
    )
    myspec = architecture_from_config(
        args.myspec_config,
        name="MySpec",
        kind="myspec",
    )
    dflash = architecture_from_config(
        args.dflash_config,
        name="DFlash",
        kind="dspark",
    )
    new_context_tokens = (
        args.new_context_tokens
        if args.new_context_tokens is not None
        else myspec.block_size + 1
    )
    report = render_markdown_report(
        title="MySpec vs DFlash FLOPs",
        contexts=args.context_lengths,
        dims=dims,
        candidate=myspec,
        baseline=dflash,
        candidate_config=args.myspec_config.resolve(),
        baseline_config=args.dflash_config.resolve(),
        new_context_tokens=new_context_tokens,
    )
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
