"""Render the final model evaluation as a care-team-facing markdown report.

Reads `src/models/final_model_report.json` (produced by `finalize.py`) and writes
`governance/MODEL_EVALUATION.md` — the operating threshold, the cost decision rule, the
honest held-out test metrics, and the threshold sensitivity. The API serves this file as
the "Model Evaluation" tab at /governance (it does not build the report itself).

Run (after finalize.py has written the JSON):
  uv run python -m src.governance.eval_report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("eval_report")

ROOT = Path(__file__).resolve().parents[2]
MODEL_REPORT = ROOT / "src/models/final_model_report.json"
OUT = ROOT / "governance" / "MODEL_EVALUATION.md"


def build_markdown(r: dict) -> str:
    cm, t, ta = r["cost_model"], r["test"], r["test"]["at_threshold"]
    thr = r["threshold"]

    def pct(x: float) -> str:
        return f"{x:.1%}"

    lines = [
        "# Model Evaluation Report",
        "",
        f"**Model:** {r['model']} · **Operating threshold:** {thr} "
        f"(flag a patient when risk ≥ {thr}).",
        "",
        "## How the care team uses this",
        "At discharge, each patient gets a **calibrated 30-day readmission risk** (an honest "
        "probability). The team proactively follows up with every patient **at or above the "
        f"flag threshold ({thr})**. The threshold is not arbitrary — it falls out of the cost "
        "trade-off below, and each prediction comes with its top contributing factors so a "
        "clinician can sanity-check *why* before acting.",
        "",
        "## The decision rule (cost assumptions)",
        f"- A 30-day readmission costs about **${cm['readmission_cost']:,}**.",
        f"- A follow-up intervention costs about **${cm['intervention_cost']:,}**.",
        f"- About **{pct(cm['preventable_fraction'])}** of readmissions are preventable "
        "with timely follow-up.",
        "",
        "Flag when the expected *preventable* cost beats the intervention cost:",
        "",
        f"&nbsp;&nbsp;`risk × {cm['preventable_fraction']} × ${cm['readmission_cost']:,} "
        f"> ${cm['intervention_cost']:,}`  →  **risk ≥ {thr}**.",
        "",
        "## Performance on held-out test data",
        "Computed **once**, on patients the model never saw during training or tuning — the "
        "honest estimate of real-world performance.",
        "",
        "| metric | value | what it means |",
        "|---|---|---|",
        f"| Recall @ {thr} | **{pct(ta['recall'])}** | share of true 30-day readmissions the model flags |",
        f"| Precision @ {thr} | {pct(ta['precision'])} | share of flagged patients who are readmitted (low by design) |",
        f"| Flag rate | {pct(ta['flag_rate'])} | share of patients recommended for follow-up |",
        f"| PR-AUC | {t['test_pr_auc']:.3f} | ranking quality at ~11% prevalence (vs 0.11 baseline) |",
        f"| ROC-AUC | {t['test_roc_auc']:.3f} | overall ranking quality |",
        f"| Brier score | {t['test_brier']:.3f} | calibration error — lower is better (0.101 = base-rate bar) |",
        f"| Net savings | **${ta['net_savings_per_1000']:,.0f} / 1,000 patients** | under the cost model above |",
        "",
        "**Why precision is low on purpose:** a wasted follow-up call costs "
        f"~${cm['intervention_cost']:,}; a *missed* readmission costs ~${cm['readmission_cost']:,}. "
        "We deliberately favour catching readmissions (high recall) over avoiding false alarms.",
        "",
        "## Threshold sensitivity",
        "The flag threshold shifts if the cost assumptions change — useful when adapting the "
        "model to a different unit or budget:",
        "",
        "| preventable fraction | readmission cost | optimal threshold |",
        "|---|---|---|",
    ]
    for s in r.get("sensitivity", []):
        lines.append(f"| {pct(s['preventable'])} | ${s['readmission_cost']:,} | {s['threshold']} |")
    lines += [
        "",
        f"The shipped threshold (**{thr}**) corresponds to the highlighted assumptions "
        f"({pct(cm['preventable_fraction'])} preventable, ${cm['readmission_cost']:,} cost).",
        "",
        "*Generated from `src/models/final_model_report.json` by "
        "`src/governance/eval_report.py`.*",
    ]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    if not MODEL_REPORT.exists():
        raise SystemExit(f"{MODEL_REPORT} not found — run src.models.finalize first.")
    report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build_markdown(report), encoding="utf-8")
    log.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
