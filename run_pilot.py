import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class EvalRecord:
    item_id: str
    category: str
    is_harmful: bool
    texts: Dict[str, str]


@dataclass
class ScoredItem:
    model_name: str
    item_id: str
    category: str
    slice_name: str
    is_harmful: bool
    score: float
    pred_default: bool
    baseline_pred_default: bool
    flipped_vs_baseline: bool
    text: str
    label_scores: Dict[str, float]


class HFTextSafetyScorer:
    def __init__(
        self,
        model_name: str,
        default_threshold: float = 0.5,
        agg_mode: str = "primary",
    ) -> None:
        self.model_name = model_name
        self.default_threshold = default_threshold
        self.agg_mode = agg_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def _extract_score(self, label_scores: Dict[str, float]) -> float:
        if not label_scores:
            return 0.0
        if self.agg_mode == "max":
            return max(label_scores.values())
        for candidate in ("toxic", "toxicity", "hate", "hate_speech"):
            if candidate in label_scores:
                return label_scores[candidate]
        return max(label_scores.values())

    @torch.inference_mode()
    def score(self, text: str) -> Tuple[float, Dict[str, float]]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)
        logits = self.model(**inputs).logits[0]
        probs = torch.sigmoid(logits).cpu().tolist()

        label_scores = {
            self.id2label.get(i, f"label_{i}"): round(float(p), 4)
            for i, p in enumerate(probs)
        }
        selected = self._extract_score(label_scores)
        return round(float(selected), 4), label_scores


def load_jsonl(path: Path) -> List[EvalRecord]:
    records: List[EvalRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            records.append(
                EvalRecord(
                    item_id=str(row.get("id", line_num)),
                    category=str(row.get("category", "uncategorized")),
                    is_harmful=bool(row["is_harmful"]),
                    texts=dict(row["texts"]),
                )
            )
    return records


def compute_slice_metrics(
    scored_items: List[ScoredItem],
    slice_name: str,
    threshold: float,
) -> Dict[str, Any]:
    subset = [x for x in scored_items if x.slice_name == slice_name]
    base_by_id = {x.item_id: x for x in scored_items if x.slice_name == "standard_en"}

    tp = fp = tn = fn = 0
    flips = 0
    scores: List[float] = []

    for item in subset:
        pred = item.score >= threshold
        scores.append(item.score)

        if item.is_harmful and pred:
            tp += 1
        elif not item.is_harmful and pred:
            fp += 1
        elif not item.is_harmful and not pred:
            tn += 1
        else:
            fn += 1

        if slice_name != "standard_en" and item.item_id in base_by_id:
            base_pred = base_by_id[item.item_id].score >= threshold
            if pred != base_pred:
                flips += 1

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "n": len(subset),
        "fnr": round(fnr, 3),
        "fpr": round(fpr, 3),
        "flips_vs_standard_en": flips if slice_name != "standard_en" else 0,
        "mean_score": round(mean_score, 4),
    }


def evaluate_model(
    records: List[EvalRecord],
    scorer: HFTextSafetyScorer,
    thresholds: List[float],
) -> Tuple[List[ScoredItem], Dict[str, Dict[str, Any]], Dict[float, Dict[str, Dict[str, Any]]]]:
    slices = list(records[0].texts.keys()) if records else []
    raw_scored: List[ScoredItem] = []

    temp_map: Dict[Tuple[str, str], Tuple[float, Dict[str, float], bool]] = {}
    for r in records:
        for s, txt in r.texts.items():
            sc, lbl_scores = scorer.score(txt)
            pred_def = sc >= scorer.default_threshold
            temp_map[(r.item_id, s)] = (sc, lbl_scores, pred_def)

    for r in records:
        base_pred_def = temp_map.get((r.item_id, "standard_en"), (0.0, {}, False))[2]
        for s, txt in r.texts.items():
            sc, lbl_scores, pred_def = temp_map[(r.item_id, s)]
            flipped = (s != "standard_en") and (pred_def != base_pred_def)
            raw_scored.append(
                ScoredItem(
                    model_name=scorer.model_name,
                    item_id=r.item_id,
                    category=r.category,
                    slice_name=s,
                    is_harmful=r.is_harmful,
                    score=sc,
                    pred_default=pred_def,
                    baseline_pred_default=base_pred_def,
                    flipped_vs_baseline=flipped,
                    text=txt,
                    label_scores=lbl_scores,
                )
            )

    default_summary = {
        s: compute_slice_metrics(raw_scored, s, scorer.default_threshold) for s in slices
    }
    sweep_summary: Dict[float, Dict[str, Dict[str, Any]]] = {
        th: {s: compute_slice_metrics(raw_scored, s, th) for s in slices}
        for th in thresholds
    }
    return raw_scored, default_summary, sweep_summary


def write_markdown_report(
    out_path: Path,
    model_results: List[Tuple[str, Dict[str, Dict[str, Any]], Dict[float, Dict[str, Dict[str, Any]]]]],
    num_records: int,
    num_slices: int,
    threshold: float,
    agg_mode: str,
    error_count: int,
) -> None:
    total_inputs = num_records * num_slices
    lines: List[str] = [
        "# Pilot Evaluation: Multilingual & Dialectal Toxicity / Hate-Speech Detection",
        "",
        f"- **Dataset size:** {num_records} paired intents × {num_slices} slices (`standard_en`, `dialect_en`, `low_res`) = {total_inputs} inputs per model",
        f"- **Active threshold / aggregation:** `{threshold}` / `{agg_mode}`",
        f"- **Total errors logged at active threshold ({threshold}):** {error_count} (see `pilot_errors.jsonl`)",
        "",
    ]

    for model_name, default_summary, sweep_summary in model_results:
        lines.append(f"## Model: `{model_name}` (Threshold = {threshold}, Agg = `{agg_mode}`)")
        lines.append("")
        lines.append("| Slice | N | FNR | FPR | Flips vs. Standard EN | Mean Score |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for s, m in default_summary.items():
            lines.append(
                f"| `{s}` | {m['n']} | {m['fnr']} | {m['fpr']} | {m['flips_vs_standard_en']} | {m['mean_score']} |"
            )
        lines.append("")

        lines.append("### Threshold Sweep (FNR / FPR by Slice)")
        lines.append("")
        lines.append("| Threshold | `standard_en` (FNR/FPR) | `dialect_en` (FNR/FPR) | `low_res` (FNR/FPR) |")
        lines.append("| :---: | :---: | :---: | :---: |")
        for th, by_slice in sweep_summary.items():
            se = by_slice.get("standard_en", {})
            de = by_slice.get("dialect_en", {})
            lr = by_slice.get("low_res", {})
            lines.append(
                f"| `{th:.1f}` | {se.get('fnr', 0)} / {se.get('fpr', 0)} | "
                f"{de.get('fnr', 0)} / {de.get('fpr', 0)} | "
                f"{lr.get('fnr', 0)} / {lr.get('fpr', 0)} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multilingual/dialectal safety pilot.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)")
    parser.add_argument(
        "--agg",
        choices=["primary", "max"],
        default="primary",
        help="Score aggregation mode: primary head vs max across all heads",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "pilot_data.jsonl"
    errors_path = base_dir / "pilot_errors.jsonl"
    report_path = base_dir / "pilot_summary.md"

    records = load_jsonl(data_path)
    num_slices = len(records[0].texts) if records else 0
    models = [
        "Horizon-Labs/multilingual-toxicity-small",
        "Horizon-Labs/multilingual-hate-speech-base",
    ]
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    all_errors: List[Dict[str, Any]] = []
    model_results = []

    for m_name in models:
        print(f"Loading and evaluating {m_name} (threshold={args.threshold}, agg={args.agg})...")
        scorer = HFTextSafetyScorer(
            model_name=m_name,
            default_threshold=args.threshold,
            agg_mode=args.agg,
        )
        scored_items, default_summary, sweep_summary = evaluate_model(records, scorer, thresholds)
        model_results.append((m_name, default_summary, sweep_summary))

        for item in scored_items:
            is_err = (item.is_harmful and not item.pred_default) or (
                (not item.is_harmful) and item.pred_default
            )
            if is_err:
                err_type = "false_negative" if item.is_harmful else "false_positive"
                row = asdict(item)
                row["error_type"] = err_type
                all_errors.append(row)

    with errors_path.open("w", encoding="utf-8") as ef:
        for err in all_errors:
            ef.write(json.dumps(err, ensure_ascii=False) + "\n")

    write_markdown_report(
        report_path,
        model_results,
        num_records=len(records),
        num_slices=num_slices,
        threshold=args.threshold,
        agg_mode=args.agg,
        error_count=len(all_errors),
    )
    print(f"\nDone. Wrote {len(all_errors)} errors to {errors_path.name} and summary to {report_path.name}.")


if __name__ == "__main__":
    main()