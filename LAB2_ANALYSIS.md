# Lab 2 ReVA VQA Analysis

## Experimental setup

Base Qwen, LoRA-fine-tuned Qwen, and VILA were evaluated on the same fixed 1,000-question ReVA subset sampled with seed 42. The subset was split into four shards of 250 questions for parallel inference. Therefore, the comparison is fair within this subset, but it is not a claim about the complete ReVA test set.

## Overall results

| Model | Questions answered | Correct | Accuracy |
|---|---:|---:|---:|
| Qwen3-VL-4B base | 1000 / 1000 | 624 | 62.40% |
| Qwen3-VL-4B + LoRA | 1000 / 1000 | 613 | 61.30% |
| VILA1.5-3B | 1000 / 1000 | 508 | 50.80% |

Base Qwen achieved the highest overall accuracy. Fine-tuned Qwen was 1.10 percentage points lower, while VILA was 11.60 points below base Qwen. The LoRA run therefore did not improve overall performance under this training configuration. Possible reasons include limited training data or duration and overfitting to patterns that did not generalize to the evaluation subset.

Temporal Grounding was difficult for every model: base Qwen scored 40.59%, fine-tuned Qwen scored 40.00%, and VILA scored 32.35%. Qwen's strongest categories were Causation Reasoning and Consequence Reasoning. VILA performed best on General Understanding, where it scored 86.96%, higher than both Qwen variants at 78.26%.

## Three qualitative examples

### 1. All three models correct — `QA-000006`

Question: How does the visible area of the parking lot change throughout the video?

Ground truth: A. Base Qwen, fine-tuned Qwen, and VILA all predicted A. This shows a case where all three pipelines successfully captured a visually observable change across the video.

### 2. Fine-tuning improvement — `QA-000791`

Question: What is the camera viewpoint classification for the video?

Ground truth: C. Base Qwen predicted D, while fine-tuned Qwen and VILA predicted C. This is an example where LoRA fine-tuning corrected a base-model error on perspective and viewpoint understanding.

### 3. Fine-tuning regression — `QA-000919`

Question: What is the primary spatial arrangement pattern of the lotus leaves in the pond?

Ground truth: D. Base Qwen and VILA predicted D, but fine-tuned Qwen predicted C. This demonstrates that fine-tuning improved some individual questions but also introduced regressions, consistent with its slightly lower overall accuracy.

## Conclusion

The data preparation, Qwen inference and scoring, LoRA fine-tuning, VILA evaluation, and model comparison pipelines all completed successfully on 1,000 questions. Base Qwen performed best overall. Because fine-tuning produced mixed improvements and regressions and reduced accuracy by 1.10 points, stronger conclusions would require additional training configurations, multiple seeds, and evaluation on the full benchmark.
