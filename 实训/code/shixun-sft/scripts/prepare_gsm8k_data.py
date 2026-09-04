#!/usr/bin/env python3
"""Convert local GSM8K parquet files to LLaMA-Factory alpaca-style JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PROMPT = (
    "Solve the following grade-school math problem. Show the reasoning steps, "
    "then put the final numeric answer after ####.\n\nProblem:\n{question}"
)


def read_parquet(path: Path) -> list[dict[str, str]]:
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    except Exception as pandas_error:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            return table.to_pylist()
        except Exception as arrow_error:
            raise RuntimeError(
                "Failed to read parquet. Please make sure pandas or pyarrow is installed in the active environment."
            ) from arrow_error


def convert_rows(rows: list[dict[str, str]], prompt_template: str) -> list[dict[str, str]]:
    converted = []
    for row in rows:
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        converted.append(
            {
                "instruction": prompt_template.format(question=question),
                "input": "",
                "output": answer,
            }
        )
    return converted


def save_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k_dir", type=Path, default=Path("/root/siton-data-1f55405a64d24fe2819a81c90df30517/20260617/gsm8k"))
    parser.add_argument("--output_dir", type=Path, default=Path("/root/siton-tmp/20235947/shixun-sft/data"))
    parser.add_argument("--train_limit", type=int, default=0, help="0 means use the full train split.")
    parser.add_argument("--test_limit", type=int, default=0, help="0 means use the full test split.")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT)
    args = parser.parse_args()

    train_path = args.gsm8k_dir / "main" / "train-00000-of-00001.parquet"
    test_path = args.gsm8k_dir / "main" / "test-00000-of-00001.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train parquet: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test parquet: {test_path}")

    train_rows = read_parquet(train_path)
    test_rows = read_parquet(test_path)

    if args.train_limit > 0:
        train_rows = train_rows[: args.train_limit]
    if args.test_limit > 0:
        test_rows = test_rows[: args.test_limit]

    save_json(args.output_dir / "gsm8k_train.json", convert_rows(train_rows, args.prompt_template))
    save_json(args.output_dir / "gsm8k_test.json", convert_rows(test_rows, args.prompt_template))

    dataset_info = {
        "gsm8k_train": {"file_name": "gsm8k_train.json"},
        "gsm8k_test": {"file_name": "gsm8k_test.json"},
    }
    save_json(args.output_dir / "dataset_info.json", dataset_info)

    print(f"Wrote {len(train_rows)} train examples to {args.output_dir / 'gsm8k_train.json'}")
    print(f"Wrote {len(test_rows)} test examples to {args.output_dir / 'gsm8k_test.json'}")
    print(f"Wrote dataset registry to {args.output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
