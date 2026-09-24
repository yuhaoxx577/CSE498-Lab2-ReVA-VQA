#!/usr/bin/env python3
"""Convert ReVA annotations into Qwen-VL conversation training data.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_video_path(file_path: str, video_root: Path | None = None) -> str:
    path = Path(file_path)
    if video_root is None:
        return path.as_posix()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(video_root).as_posix()
    except ValueError:
        return path.as_posix()


def format_options(options: dict[str, str]) -> str:
    """Return sorted multiple-choice options, one option per line."""
    return "\n".join(
        f"{label}. {options[label]}" for label in sorted(options)
    )


def format_question(question: str, options: dict[str, str], prompt_style: str) -> str:
    """Build the human message used by Qwen SFT."""
    option_text = format_options(options)

    if prompt_style == "plain":
        return (
            f"<video>\n"
            f"Question: {question}\n"
            f"{option_text}"
        )

    if prompt_style == "reva_eval":
        return (
            f"<video>\n"
            f"Please carefully watch the video and answer the following question.\n"
            f"Question: {question}\n"
            f"{option_text}\n"
            f"Put the option letter inside <answer> </answer>."
        )

    raise ValueError(f"Unknown prompt style: {prompt_style}")


def format_answer(qa: dict[str, Any], answer_style: str) -> str:
    """Format the assistant target from a ReVA QA item."""
    answer = str(qa["correct_answer"]).strip().upper()

    if answer_style == "letter":
        return f"Answer: {answer}"

    if answer_style =="tagged":
        return f"<answer>{answer}</answer>"

    if answer_style =="cot_tagged":
        reasoning = str(qa.get("reasoning", "")).strip()
        return f"<think>{reasoning}</think><answer>{answer}</answer>"

    raise ValueError(f"Unknown answer style: {answer_style}")


def iter_reva_qas(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one flat QA record at a time from nested ReVA annotations."""
    for video_id, video_data in data["videos"].items():
        for category, question_groups in video_data["mcq"].items():
            for question_type, qa_list in question_groups.items():
                for qa in qa_list:
                    yield {
                        **qa,
                        "video_id": video_id,
                        "file_path": video_data["file_path"],
                        "dataset_name": video_data.get("dataset_name", ""),
                        "category": category,
                        "question_type": question_type,
                    }


def convert_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """Convert one flat ReVA QA item into one Qwen conversation sample."""
    video_path = normalize_video_path(item["file_path"], args.video_root)

    if args.require_video:
        full_video_path = args.video_root / video_path
        if not full_video_path.exists():
            return None

    question_text = format_question(
        item["question"],
        item["options"],
        args.prompt_style,
    )

    answer_text = format_answer(item, args.answer_style)

    sample = {
        "video": video_path,
        "conversations": [
            {
                "from": "human",
                "value": question_text,
            },
            {
                "from": "gpt",
                "value": answer_text,
            },
        ],
    }

    if args.keep_metadata:
        sample["metadata"] = {
            "video_id": item.get("video_id", ""),
            "qa_id": item.get("qa_id", ""),
            "dataset_name": item.get("dataset_name", ""),
            "category": item.get("category", ""),
            "question_type": item.get("question_type", "")
        }

        return sample

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/reva_train/train_set.json"))
    parser.add_argument("--output", type=Path, default=Path("data/qwen_train/train.json"))
    parser.add_argument("--video-root", type=Path, default=Path("data/qwen_train"))
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all.")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all.")
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--prompt-style", choices=["plain", "reva_eval"], default="reva_eval")
    parser.add_argument("--answer-style", choices=["letter", "tagged", "cot_tagged"], default="tagged")
    parser.add_argument("--keep-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)

    samples = []
    seen_videos = set()
    skipped_missing_video = 0
    for item in iter_reva_qas(data):
        seen_videos.add(item["video_id"])
        if args.max_videos and len(seen_videos) > args.max_videos:
            break
        sample = convert_item(item, args)
        if sample is None:
            skipped_missing_video += 1
            continue
        samples.append(sample)
        if args.max_samples and len(samples) >= args.max_samples:
            break

    if args.require_video and not samples:
        raise SystemExit(
            "No training samples reference an existing video. "
            "Check QWEN_VIDEO_ROOT and the downloaded ReVA directory; output was not modified."
        )

    dump_json(samples, args.output)
    print(f"Saved {len(samples)} Qwen training samples to {args.output}")
    if skipped_missing_video:
        print(f"Skipped {skipped_missing_video} samples with missing videos")


if __name__ == "__main__":
    main()
