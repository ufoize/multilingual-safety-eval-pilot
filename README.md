# Multilingual & Dialectal Safety Evaluation Pilot

A lightweight, reproducible evaluation harness testing how open-weights multilingual toxicity and hate-speech classifiers behave across **Standard English (`standard_en`)**, **Dialectal/Colloquial English (`dialect_en`)**, and **Low-Resource / Romanized Urdu-Hindi (`low_res`)** inputs.

## Motivation
Safety guardrails and content moderation classifiers are typically calibrated on high-resource standard English. When deployed across dialectal varieties, code-switched phrasing, or Romanized low-resource languages, threshold calibration and label-head mismatches can lead to elevated false-negative rates (missed harm) or false-positive flips on benign slang and technical terms.

## Models Evaluated
- `Horizon-Labs/multilingual-toxicity-small`
- `Horizon-Labs/multilingual-hate-speech-base`

## Repository Contents
- `pilot_data.jsonl`: Paired evaluation dataset (24 intents × 3 linguistic slices = 72 inputs per model) covering both harmful and benign edge cases.
- `run_pilot.py`: Inference and evaluation script supporting configurable decision thresholds (`--threshold`), label aggregation modes (`--agg primary|max`), slice-level FNR/FPR metrics, prediction flip tracking relative to `standard_en`, and threshold sweeps (`0.2`–`0.7`).
- `pilot_summary.md`: Generated Markdown report summarizing slice-level metrics and threshold sweeps.

## Quickstart

```bash

### CLI Options
- `--threshold` (default: `0.5`): Decision threshold for binary harmful/non-harmful classification and flip detection.
- `--agg` (`primary` or `max`, default: `primary`):
  - `primary`: Uses the model's main `toxicity` / `hate` head.
  - `max`: Uses the maximum probability across all output heads (e.g., `threat`, `insult`, `identity_hostility`).
pip install torch transformers
python run_pilot.py --threshold 0.3 --agg primary
