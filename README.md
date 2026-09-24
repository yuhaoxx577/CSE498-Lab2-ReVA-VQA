# CSE 498 Lab 2 — ReVA Video Question Answering

Student submission for evaluating Qwen3-VL and VILA on the ReVA VQA benchmark and fine-tuning Qwen3-VL with LoRA.

## Submission contents

- Completed student code:
  - `scripts/build_qwen_train_data.py`
  - `reva_eval/data/rsvidqa/prepare_reva_v2_test_set.py`
  - `scripts/score_reva_predictions.py`
  - `vila_eval/reva_v2.py`
- Qwen LoRA training and evaluation scripts
- Unit tests supplied with the assignment
- Results from the same fixed 1,000-question subset for all three models
- Qualitative analysis and comparison charts

The full ReVA dataset and model/checkpoint weights are intentionally excluded because they are large external resources. Small demo data is included so the preprocessing pipeline can be inspected without downloading the full dataset.

## Final results

| Model | Correct | Accuracy |
|---|---:|---:|
| Qwen3-VL-4B base | 624 / 1000 | 62.40% |
| Qwen3-VL-4B + LoRA | 613 / 1000 | 61.30% |
| VILA1.5-3B | 508 / 1000 | 50.80% |

All three models answered all 1,000 questions. The fixed evaluation subset was sampled with seed 42 and split into four 250-question shards for parallel inference.

Detailed results:

- [`LAB2_ANALYSIS.md`](LAB2_ANALYSIS.md)
- [`outputs/final_results/model_comparison.csv`](outputs/final_results/model_comparison.csv)
- [`outputs/qwen_base_1000/qwen_base_1000/result.csv`](outputs/qwen_base_1000/qwen_base_1000/result.csv)
- [`outputs/qwen_finetuned_1000/qwen_finetuned_1000/result.csv`](outputs/qwen_finetuned_1000/qwen_finetuned_1000/result.csv)
- [`outputs/vila_1000/metrics.json`](outputs/vila_1000/metrics.json)

## Validation

Run the assignment tests from the repository root:

```bash
python3 -m pytest -q
```

The tests exercise data conversion, stable question IDs, answer parsing, missing predictions, VILA prompt construction, and accuracy summaries without loading a large model.

For dataset/model setup and the complete GPU commands, see [`ASSIGNMENT.md`](ASSIGNMENT.md) and [`STUDENT_GUIDE.md`](STUDENT_GUIDE.md).
