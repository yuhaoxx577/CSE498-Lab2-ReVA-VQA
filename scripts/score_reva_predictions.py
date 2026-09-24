#!/usr/bin/env python3
"""Score Qwen-style ReVA prediction JSONL files.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_answer(text: str) -> str:
    """Extract text inside <answer>...</answer>; fall back to full text."""
    text = str(text)

    match = re.search(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return text.strip()


def extract_letter(text: str) -> str:
    """Extract one answer letter A-H from model output."""
    answer_text = extract_answer(text).strip().upper()

    if re.fullmatch(r"[A-H]", answer_text):
        return answer_text

    patterns = [
        r"\bFINAL\s+ANSWER\s*[:\-]?\s*([A-H])\b",
        r"\bANSWER\s+(?:IS\s+)?[:\-]?\s*([A-H])\b",
        r"\bOPTION\s+([A-H])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, answer_text)
        if match:
            return match.group(1)

    letters = re.findall(r"\b[A-H]\b", answer_text)

    if letters:
        return letters[-1]

    return ""


def load_predictions(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL files from output_dir, ignoring result.json."""
    predictions = {}

    for file_path in sorted(output_dir.glob("*.json")):
        if file_path.name == "result.json":
            continue

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)
                item_id = str(item["id"])
                predictions[item_id] = item

    return predictions


def score_predictions(preds: dict[str, dict[str, Any]],gt_list: list[dict[str, Any]],) -> tuple[dict, str]:
    """Score every GT item and report completion separately from accuracy."""

    results = {}
    completed = 0
    correct = 0

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for gt_item in gt_list:
        qa_id = str(gt_item["id"])
        correct_letter = str(gt_item["answer"]).strip().upper()
        question_type = str(gt_item.get("question_type", "unknown"))

        pred_item = preds.get(qa_id)

        if pred_item is None:
            pred_text = ""
            pred_letter = ""
        else:
            completed += 1
            pred_text = str(pred_item.get("pred", ""))
            pred_letter = extract_letter(pred_text)

        acc = int(pred_letter == correct_letter)
        correct += acc

        category_stats[question_type]["correct"] += acc
        category_stats[question_type]["total"] += 1

        results[qa_id] = {
            "id": qa_id,
            "pred": pred_text,
            "pred_letter": pred_letter,
            "answer": correct_letter,
            "question_type": question_type,
            "acc": acc,
        }

    total = len(gt_list)

    completed_percent = 100 * completed / total if total else 0.0
    accuracy_percent = 100 * correct / total if total else 0.0

    lines = [
        f"Completed: {completed}/{total} = {completed_percent:.2f}%",
        f"Total: {correct}/{total} = {accuracy_percent:.2f}%",
        "question_type,correct,total,accuracy",
    ]

    for question_type in sorted(category_stats):
        stats = category_stats[question_type]
        category_correct = stats["correct"]
        category_total = stats["total"]

        category_percent = (
            100 * category_correct / category_total
            if category_total
            else 0.0
        )

        lines.append(
            f"{question_type},{category_correct},"
            f"{category_total},{category_percent:.2f}%"
        )

    csv_text = "\n".join(lines) + "\n"

    return results, csv_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = load_predictions(args.output_dir)
    gt_list = json.loads(args.gt_file.read_text(encoding="utf-8"))
    results, csv_text = score_predictions(preds, gt_list)

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Results ===")
    print(csv_text)

    csv_path = args.output_dir / "result.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
