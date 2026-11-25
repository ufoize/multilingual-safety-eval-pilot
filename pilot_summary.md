# Pilot Evaluation: Multilingual & Dialectal Toxicity / Hate-Speech Detection

- **Dataset size:** 24 paired intents × 3 slices (`standard_en`, `dialect_en`, `low_res`) = 72 inputs per model
- **Total errors logged at default threshold (0.5):** 55 (see `pilot_errors.jsonl`)

## Model: `Horizon-Labs/multilingual-toxicity-small` (Default Threshold = 0.5)

| Slice | N | FNR | FPR | Flips vs. Standard EN | Mean Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `standard_en` | 24 | 0.667 | 0.0 | 0 | 0.2084 |
| `dialect_en` | 24 | 0.75 | 0.083 | 2 | 0.2116 |
| `low_res` | 24 | 0.667 | 0.0 | 2 | 0.2618 |

### Threshold Sweep (FNR / FPR by Slice)

| Threshold | `standard_en` (FNR/FPR) | `dialect_en` (FNR/FPR) | `low_res` (FNR/FPR) |
| :---: | :---: | :---: | :---: |
| `0.2` | 0.25 / 0.0 | 0.5 / 0.083 | 0.0 / 0.083 |
| `0.3` | 0.417 / 0.0 | 0.667 / 0.083 | 0.167 / 0.0 |
| `0.4` | 0.5 / 0.0 | 0.75 / 0.083 | 0.583 / 0.0 |
| `0.5` | 0.667 / 0.0 | 0.75 / 0.083 | 0.667 / 0.0 |
| `0.6` | 0.75 / 0.0 | 0.75 / 0.083 | 0.833 / 0.0 |
| `0.7` | 0.833 / 0.0 | 0.75 / 0.0 | 1.0 / 0.0 |

## Model: `Horizon-Labs/multilingual-hate-speech-base` (Default Threshold = 0.5)

| Slice | N | FNR | FPR | Flips vs. Standard EN | Mean Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `standard_en` | 24 | 0.833 | 0.0 | 0 | 0.1155 |
| `dialect_en` | 24 | 0.75 | 0.0 | 1 | 0.1293 |
| `low_res` | 24 | 0.833 | 0.0 | 2 | 0.1288 |

### Threshold Sweep (FNR / FPR by Slice)

| Threshold | `standard_en` (FNR/FPR) | `dialect_en` (FNR/FPR) | `low_res` (FNR/FPR) |
| :---: | :---: | :---: | :---: |
| `0.2` | 0.667 / 0.0 | 0.667 / 0.0 | 0.667 / 0.0 |
| `0.3` | 0.667 / 0.0 | 0.75 / 0.0 | 0.667 / 0.0 |
| `0.4` | 0.75 / 0.0 | 0.75 / 0.0 | 0.75 / 0.0 |
| `0.5` | 0.833 / 0.0 | 0.75 / 0.0 | 0.833 / 0.0 |
| `0.6` | 0.833 / 0.0 | 0.75 / 0.0 | 0.917 / 0.0 |
| `0.7` | 0.917 / 0.0 | 0.833 / 0.0 | 1.0 / 0.0 |
