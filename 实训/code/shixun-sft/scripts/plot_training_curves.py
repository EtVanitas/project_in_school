#!/usr/bin/env python3
"""Plot LLaMA-Factory Trainer logs without online tracking services."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def smooth(values: list[float]) -> list[float]:
    if not values:
        return []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(values))) - 0.5)
    last = values[0]
    smoothed = []
    for value in values:
        last = last * weight + (1 - weight) * value
        smoothed.append(last)
    return smoothed


def collect(log_history: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps, values = [], []
    for item in log_history:
        if key in item and "step" in item and item[key] is not None:
            steps.append(int(item["step"]))
            values.append(float(item[key]))
    return steps, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=Path("/root/siton-tmp/20235947/shixun-sft/outputs/qwen15_gsm8k_full_sft/trainer_state.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("/root/siton-tmp/20235947/shixun-sft/outputs/figures"))
    parser.add_argument("--keys", nargs="*", default=["loss", "grad_norm", "learning_rate", "eval_loss", "eval_accuracy"])
    args = parser.parse_args()

    if not args.state.exists():
        raise FileNotFoundError(f"Missing trainer state: {args.state}")

    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        raise RuntimeError("matplotlib is required for plotting training curves.") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.state.open(encoding="utf-8") as f:
        log_history = json.load(f)["log_history"]

    csv_path = args.output_dir / "trainer_log_metrics.csv"
    all_keys = sorted({key for row in log_history for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(log_history)

    plotted = []
    for key in args.keys:
        steps, values = collect(log_history, key)
        if not values:
            print(f"Skip {key}: no values in trainer_state.json")
            continue

        plt.figure(figsize=(7, 4))
        plt.plot(steps, values, alpha=0.35, label="original")
        if len(values) > 2:
            plt.plot(steps, smooth(values), label="smoothed")
        plt.xlabel("step")
        plt.ylabel(key)
        plt.title(key)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        figure_path = args.output_dir / f"{key.replace('/', '_')}.png"
        plt.savefig(figure_path, dpi=160)
        plt.close()
        plotted.append(str(figure_path))
        print(f"Saved {figure_path}")

    if plotted:
        print(f"Saved metric csv to {csv_path}")


if __name__ == "__main__":
    main()
