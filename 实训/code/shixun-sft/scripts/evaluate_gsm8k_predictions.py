#!/usr/bin/env python3
"""Evaluate GSM8K generated_predictions.jsonl with exact numeric match."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
HASH_ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def normalize_number(text: str) -> str | None:
    if text is None:
        return None
    text = str(text)
    match = HASH_ANSWER_RE.search(text)
    if match:
        raw = match.group(1)
    else:
        matches = NUMBER_RE.findall(text)
        if not matches:
            return None
        raw = matches[-1]

    raw = raw.replace(",", "")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return raw
    return format(value.normalize(), "f").rstrip("0").rstrip(".") if "." in format(value, "f") else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("/root/siton-tmp/20235947/shixun-sft/outputs/qwen15_gsm8k_full_predict/generated_predictions.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("/root/siton-tmp/20235947/shixun-sft/outputs/eval"))
    args = parser.parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(f"Missing predictions file: {args.predictions}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    correct = 0
    total = 0
    with args.predictions.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            pred_answer = normalize_number(item.get("predict", ""))
            label_answer = normalize_number(item.get("label", ""))
            is_correct = pred_answer is not None and label_answer is not None and pred_answer == label_answer
            total += 1
            correct += int(is_correct)
            rows.append(
                {
                    "correct": is_correct,
                    "pred_answer": pred_answer,
                    "label_answer": label_answer,
                    "predict": item.get("predict", ""),
                    "label": item.get("label", ""),
                    "prompt": item.get("prompt", ""),
                }
            )

    accuracy = correct / total if total else 0.0
    summary = {
        "predictions": str(args.predictions),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }

    summary_path = args.output_dir / "gsm8k_exact_match_metrics.json"
    errors_path = args.output_dir / "gsm8k_wrong_cases.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with errors_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if not row["correct"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote wrong cases to {errors_path}")


if __name__ == "__main__":
    main()
