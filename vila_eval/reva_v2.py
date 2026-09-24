import argparse
import copy
import json
import os
import re
from pathlib import Path
from time import strftime
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=".data/ReVA_V2/valid_set.json")
    parser.add_argument("--dataset-root", type=str, default=".data")
    parser.add_argument("--dataset-prefix", type=str, default="#dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--video-max-tiles", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timestamp-output", action="store_true")
    return parser.parse_args()


def load_instances(question_file: str) -> list[dict[str, Any]]:
    """Flatten nested ReVA annotations into VILA evaluation instances."""

    with open(question_file, encoding="utf-8") as f:
        data = json.load(f)

    instances = []

    for video_id, video_data in data["videos"].items():
        file_path = video_data["file_path"]
        dataset_name = video_data.get("dataset_name", "")

        for category, question_groups in video_data["mcq"].items():
            for subcategory, qa_list in question_groups.items():
                for question_idx, qa in enumerate(qa_list):
                    instance = copy.deepcopy(qa)

                    if qa.get("qa_id"):
                        qa_id = str(qa["qa_id"])

                    elif qa.get("global_index") is not None:
                        qa_id = f"REVA-G{int(qa['global_index']):06d}"

                    else:
                        video_stem = Path(file_path).stem
                        subcategory_code = subcategory[:3].upper()

                        qa_id = (
                            f"REVA-{video_stem}-"
                            f"{subcategory_code}-{question_idx:04d}"
                        )

                    instance["qa_id"] = qa_id
                    instance["video_id"] = video_id
                    instance["file_path"] = file_path
                    instance["dataset_name"] = dataset_name
                    instance["category"] = category
                    instance["subcategory"] = subcategory

                    instances.append(instance)

    return instances


def resolve_video_path(raw_path: str, dataset_root: str, dataset_prefix: str,) -> str:
    """Resolve a ReVA annotation path to an existing local video path."""

    normalized_path = raw_path.replace("\\", "/")

    candidates = [
        normalized_path,
        os.path.join(dataset_root, normalized_path.lstrip("/")),
    ]

    if normalized_path.startswith(dataset_prefix):
        without_prefix = normalized_path[len(dataset_prefix):].lstrip("/")

        candidates.append(
            os.path.join(dataset_root, without_prefix)
        )

        path_parts = Path(without_prefix).parts

        if path_parts and path_parts[0].lower() in {"reva_v2", "reva"}:
            candidates.append(
                os.path.join(dataset_root, *path_parts[1:])
            )

    for candidate in candidates:
        candidate = os.path.normpath(candidate)

        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find video for path: {raw_path}"
    )


def build_prompt(question: str, options: dict[str, str]) -> str:
    """Build the VILA multiple-choice prompt."""

    lines = [question.strip()]

    for letter in sorted(options):
        option_text = str(options[letter]).strip()
        lines.append(f"{letter.upper()}. {option_text}")

    lines.append(
        "Answer with only the option letter from the given choices."
    )

    return "\n".join(lines)


def parse_choice(response: str, options: dict[str, str],) -> str | None:
    """Parse a choice letter from VILA raw response."""

    response = str(response).strip()

    if not response:
        return None

    valid_letters = {
        str(letter).upper()
        for letter in options
    }

    upper_response = response.upper()

    direct_match = re.fullmatch(
        r"[\(\[]?([A-Z])[\)\]\.\,\:\;\!\?]?",
        upper_response,
    )

    if direct_match:
        letter = direct_match.group(1)

        if letter in valid_letters:
            return letter

    patterns = [
        r"\bFINAL\s+ANSWER\s*(?:IS|:|-)?\s*\(?([A-Z])\)?",
        r"\bANSWER\s*(?:IS|:|-)\s*\(?([A-Z])\)?",
        r"\bOPTION\s*\(?([A-Z])\)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, upper_response)

        if match:
            letter = match.group(1)

            if letter in valid_letters:
                return letter

    normalized_response = re.sub(
        r"[^\w\s]",
        " ",
        response.lower(),
    )
    normalized_response = re.sub(
        r"\s+",
        " ",
        normalized_response,
    ).strip()

    matching_letters = []

    for letter, option_text in options.items():
        normalized_option = re.sub(
            r"[^\w\s]",
            " ",
            str(option_text).lower(),
        )
        normalized_option = re.sub(
            r"\s+",
            " ",
            normalized_option,
        ).strip()

        if normalized_option and re.search(
            rf"\b{re.escape(normalized_option)}\b",
            normalized_response,
        ):
            matching_letters.append(str(letter).upper())

    if len(matching_letters) == 1:
        return matching_letters[0]

    return None


def load_existing_predictions(output_path: str) -> dict[str, dict]:
    if not os.path.exists(output_path):
        return {}
    predictions = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions[record["qa_id"]] = record
    return predictions


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def summarize(records: list[dict]) -> dict:
    """Compute total, category, and subcategory accuracy for VILA outputs."""

    def calculate_stats(items: list[dict]) -> dict:
        num_questions = len(items)

        num_answered = sum(
            1
            for item in items
            if item.get("pred_letter") not in (None, "")
        )

        num_correct = sum(
            1
            for item in items
            if bool(item.get("is_correct", False))
        )

        accuracy = (
            num_correct / num_questions
            if num_questions > 0
            else 0.0
        )

        return {
            "num_questions": num_questions,
            "num_answered": num_answered,
            "num_correct": num_correct,
            "accuracy": accuracy,
        }

    category_records = {}
    subcategory_records = {}

    for record in records:
        category = str(record.get("category", "unknown"))
        subcategory = str(record.get("subcategory", "unknown"))

        category_records.setdefault(category, []).append(record)
        subcategory_records.setdefault(subcategory, []).append(record)

    by_category = {
        category: calculate_stats(items)
        for category, items in category_records.items()
    }

    by_subcategory = {
        subcategory: calculate_stats(items)
        for subcategory, items in subcategory_records.items()
    }

    overall = calculate_stats(records)

    return {
        "num_questions": overall["num_questions"],
        "num_answered": overall["num_answered"],
        "num_correct": overall["num_correct"],
        "accuracy": overall["accuracy"],
        "by_category": by_category,
        "by_subcategory": by_subcategory,
    }

def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import llava
    from llava import Video
    from llava import conversation as conversation_lib

    instances = load_instances(args.question_file)
    if args.max_questions is not None:
        instances = instances[: args.max_questions]

    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir if args.resume or not args.timestamp_output else base_output_dir / strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation artifacts to: {output_dir}")

    outputs_path = output_dir / "outputs.jsonl"
    metrics_path = output_dir / "metrics.json"
    existing = load_existing_predictions(str(outputs_path)) if args.resume else {}

    if args.conv_mode != "auto":
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    model = llava.load(args.model_path, model_base=args.model_base)
    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames
    if args.video_max_tiles > 0:
        model.config.video_max_tiles = args.video_max_tiles
        model.llm.config.video_max_tiles = args.video_max_tiles

    generation_config = copy.deepcopy(model.default_generation_config)
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    outputs = []
    for instance in tqdm(instances):
        if instance["qa_id"] in existing:
            outputs.append(existing[instance["qa_id"]])
            continue

        video_path = resolve_video_path(instance["file_path"], args.dataset_root, args.dataset_prefix)
        prompt = build_prompt(instance["question"], instance["options"])
        response = model.generate_content([Video(video_path), prompt], generation_config=generation_config)
        pred_letter = parse_choice(response, instance["options"])

        outputs.append({
            "qa_id": instance["qa_id"],
            "video_id": instance["video_id"],
            "video_path": video_path,
            "dataset_name": instance.get("dataset_name"),
            "category": instance["category"],
            "subcategory": instance["subcategory"],
            "question": instance["question"],
            "options": instance["options"],
            "prompt": prompt,
            "raw_response": response,
            "pred_letter": pred_letter,
            "correct_answer": instance["correct_answer"],
            "is_correct": pred_letter == instance["correct_answer"],
            "reasoning": instance.get("reasoning", ""),
            "example": instance.get("example", ""),
        })
        save_jsonl(str(outputs_path), outputs)

    save_jsonl(str(outputs_path), outputs)
    save_json(str(metrics_path), summarize(outputs))


if __name__ == "__main__":
    main()
