"""Build the demo presentation (PPTX) in the project's bottle-green palette.

Numbers are pulled from the real artifacts where possible (model report) and the
documented Stage-3 comparison. Run:

  uv run python presentation/build_deck.py    # -> presentation/diabetes_readmission_demo.pptx
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "presentation"

# --- palette (matches the UI) ------------------------------------------------
ACCENT = RGBColor(0x0B, 0x6B, 0x4F)   # bottle green
BG = RGBColor(0xD7, 0xE9, 0xDD)       # light bottle green
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TEXT = RGBColor(0x16, 0x26, 0x1D)
MUTED = RGBColor(0x5A, 0x6F, 0x63)
PANEL = RGBColor(0xEE, 0xF5, 0xF0)
CHIP = RGBColor(0xBF, 0xE0, 0xCC)
RED = RGBColor(0xC0, 0x39, 0x2B)

SW, SH = Inches(13.333), Inches(7.5)
report = json.loads((ROOT / "src/models/final_model_report.json").read_text())
ta = report["test"]["at_threshold"]
t = report["test"]
cm = report["cost_model"]


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH
    return prs


def _bg(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def _box(slide, l, tp, w, h, fill=None, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, tp, w, h)
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(1)
    return shp


def title_slide(prs, title, subtitle, footer):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s, ACCENT)
    tb = s.shapes.add_textbox(Inches(0.9), Inches(2.4), Inches(11.5), Inches(2))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = title
    p.font.size = Pt(40); p.font.bold = True; p.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.text = subtitle
    p2.font.size = Pt(20); p2.font.color.rgb = CHIP
    fb = s.shapes.add_textbox(Inches(0.9), Inches(6.6), Inches(11.5), Inches(0.6))
    fp = fb.text_frame.paragraphs[0]; fp.text = footer
    fp.font.size = Pt(13); fp.font.color.rgb = CHIP
    return s


def content_slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _bg(s)
    bar = _box(s, 0, 0, SW, Inches(1.0), fill=ACCENT)
    tf = bar.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.6); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = title
    p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = WHITE
    return s


def bullets(slide, l, tp, w, h, items, size=15, title=None, title_color=ACCENT):
    tb = slide.shapes.add_textbox(l, tp, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    if title:
        p = tf.paragraphs[0]; p.text = title
        p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = title_color
        first = False
    for it in items:
        indent = it.startswith("  ")
        text = it.strip()
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = ("– " if indent else "• ") + text
        p.font.size = Pt(size - 2 if indent else size)
        p.font.color.rgb = MUTED if indent else TEXT
        p.level = 1 if indent else 0
        p.space_after = Pt(4)
    return tb


def table(slide, l, tp, w, rows, col_w, highlight=None, header_fill=ACCENT, fsize=12):
    nrows, ncols = len(rows), len(rows[0])
    h = Inches(0.36 * nrows)
    gt = slide.shapes.add_table(nrows, ncols, l, tp, w, h).table
    for j, cw in enumerate(col_w):
        gt.columns[j].width = cw
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = gt.cell(i, j)
            c.text = str(val)
            para = c.text_frame.paragraphs[0]
            para.font.size = Pt(fsize)
            para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            if i == 0:
                c.fill.solid(); c.fill.fore_color.rgb = header_fill
                para.font.color.rgb = WHITE; para.font.bold = True
            else:
                c.fill.solid()
                c.fill.fore_color.rgb = CHIP if highlight == i else PANEL
                para.font.color.rgb = TEXT
                para.font.bold = highlight == i
    return gt


def card(slide, l, tp, w, h, heading, items, size=14):
    _box(slide, l, tp, w, h, fill=WHITE, line=RGBColor(0xC2, 0xD6, 0xC9))
    bullets(slide, l + Inches(0.25), tp + Inches(0.18), w - Inches(0.5), h - Inches(0.3),
            items, size=size, title=heading)


def build():
    prs = new_deck()

    title_slide(
        prs,
        "Diabetes 30-Day Readmission Prediction",
        "End-to-end ML system — demo walkthrough",
        "Care-team decision support  ·  UCI Diabetes 130-US Hospitals (1999–2008)  ·  ~101k encounters",
    )

    # 1. EDA + column segregation
    s = content_slide(prs, "1.  EDA & Column Segregation")
    bullets(s, Inches(0.6), Inches(1.25), Inches(5.4), Inches(5.8), [
        "101,766 encounters (1 row = 1 hospital stay), ~50 columns",
        "Target: readmitted ∈ {<30, >30, NO} → binary '<30' = 1",
        "  Positive rate ~11.3% → accuracy is a trap",
        "Missing coded as '?' → treated as NaN on load",
        "weight ~97% missing → dropped (justified)",
        "A missing A1C/glucose test is information →",
        "  kept as its own 'NotMeasured' category, never imputed",
        "Excluded encounters that can't be readmitted",
        "  (expired / hospice discharge)",
    ], size=15, title="Dataset & traps")
    card(s, Inches(6.3), Inches(1.25), Inches(6.4), Inches(5.7), "Columns grouped into blocks", [
        "Demographics: race, gender, age",
        "Encounter: time_in_hospital, num_lab_procedures,",
        "  num_procedures, num_medications, number_diagnoses, medical_specialty",
        "Prior utilization: number_outpatient / emergency / inpatient, payer_code",
        "Diagnoses & labs: diag_1/2/3, max_glu_serum, A1Cresult",
        "Medications: 21 drug flags + change + diabetesMed",
        "Admission / discharge: type, source, disposition",
    ], size=13)

    # 2. Engineered features
    s = content_slide(prs, "2.  Engineered Features")
    bullets(s, Inches(0.6), Inches(1.25), Inches(12.1), Inches(5.6), [
        "ICD-9 diag_1/2/3 → clinical buckets (Strack et al.): Circulatory, Respiratory,",
        "  Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other",
        "  → diag_1_group, diag_2_group, diag_3_group, diabetes_diag_any",
        "total_prior_visits = number_outpatient + number_emergency + number_inpatient",
        "n_med_changes, n_active_meds  → summarized from the 21 medication columns",
        "age_ordinal  → age bracket mapped to an ordinal 0–9",
        "lump_rare: rare medical_specialty / payer_code (<1% of train) collapsed to 'Other'",
        "One shared engineer() runs at BOTH train and serve time → no training/serving skew",
        "Every feature + source + rationale logged in src/features/FEATURES.md",
    ], size=16)

    # 3. Features going to the model
    s = content_slide(prs, "3.  Features Going to the Model")
    bullets(s, Inches(0.6), Inches(1.25), Inches(6.0), Inches(5.6), [
        "48 model features = 13 numeric + 35 categorical",
        "Numeric: time_in_hospital, num_lab_procedures, num_procedures,",
        "  num_medications, number_outpatient/emergency/inpatient,",
        "  number_diagnoses, total_prior_visits, n_med_changes,",
        "  n_active_meds, diabetes_diag_any, age_ordinal",
        "Categorical: 21 medication flags + race, gender, labs,",
        "  admission/discharge, specialty, payer, diag groups",
    ], size=14, title="The 48-feature matrix → features.parquet")
    card(s, Inches(6.9), Inches(1.25), Inches(5.8), Inches(5.6), "26 human inputs → 48 features", [
        "Care team types ~26 facts; only gender & age required",
        "Server DERIVES ~8 engineered features (not typed)",
        "  visit counts → total_prior_visits; codes → diag groups; age → age_ordinal",
        "18 of 21 drugs default to 'No' (not prescribed)",
        "  form surfaces the common ones (metformin, glyburide, insulin)",
        "XGBoost native-categorical: category order frozen in feature_spec.json",
    ], size=13)

    # 4. Modeling comparison
    s = content_slide(prs, "4.  Modeling — Baseline vs Advanced")
    table(s, Inches(0.6), Inches(1.35), Inches(8.4), [
        ["model (validation)", "PR-AUC", "ROC-AUC", "recall@p≥30%", "Brier"],
        ["Logistic regression (baseline)", "0.221", "0.667", "0.189", "0.225"],
        ["XGBoost  (winner)", "0.239", "0.680", "0.216", "0.206"],
        ["LightGBM", "0.239", "0.675", "0.215", "0.201"],
        ["XGBoost + isotonic calibration", "0.236", "0.680", "0.205", "0.097"],
    ], col_w=[Inches(3.4), Inches(1.25), Inches(1.25), Inches(1.5), Inches(1.0)], highlight=2)
    bullets(s, Inches(0.6), Inches(4.4), Inches(12.1), Inches(2.7), [
        "Split by patient_nbr (GroupShuffleSplit); tuned with StratifiedGroupKFold → no leakage",
        "Imbalance via class weights (not resampling) — keeps grouped CV & calibration valid",
        "Tuned on PR-AUC, not ROC-AUC — at 11% positives PR-AUC is what the care team feels",
        "Calibration fixed the Brier disaster: 0.206 → 0.097 with ranking intact (honest %)",
    ], size=14)

    # 5. Model evaluation summary
    s = content_slide(prs, "5.  Model Evaluation (held-out test)")
    table(s, Inches(0.6), Inches(1.35), Inches(6.6), [
        ["metric", "value"],
        ["Recall @ 0.10", f"{ta['recall']:.1%}"],
        ["Precision @ 0.10", f"{ta['precision']:.1%}"],
        ["Flag rate", f"{ta['flag_rate']:.1%}"],
        ["PR-AUC", f"{t['test_pr_auc']:.3f}"],
        ["ROC-AUC", f"{t['test_roc_auc']:.3f}"],
        ["Brier", f"{t['test_brier']:.3f}"],
        ["Net savings / 1,000", f"${ta['net_savings_per_1000']:,.0f}"],
    ], col_w=[Inches(4.4), Inches(2.2)], highlight=1, fsize=13)
    bullets(s, Inches(0.6), Inches(5.0), Inches(6.6), Inches(2.2), [
        f"Decision rule: risk × {cm['preventable_fraction']} × ${cm['readmission_cost']:,}",
        f"  > ${cm['intervention_cost']:,}  →  flag when risk ≥ {report['threshold']}",
        "Low precision is BY DESIGN — a $300 call vs a $15k readmission",
    ], size=13)
    if (ROOT / "governance/shap_global_importance.png").exists():
        s.shapes.add_picture(str(ROOT / "governance/shap_global_importance.png"),
                             Inches(7.5), Inches(1.3), height=Inches(5.5))

    # 6. Monitoring
    s = content_slide(prs, "6.  Monitoring — What & How")
    card(s, Inches(0.6), Inches(1.25), Inches(6.0), Inches(5.7), "Service & model metrics", [
        "Prometheus scrapes /metrics every 15s → Grafana",
        "  request rate, p95 latency, error rate, flag rate, score distribution",
        "Every prediction logged to logs/predictions.jsonl",
        "  inputs + score + version + timestamp (audit + drift window)",
        "Alerts (Prometheus rules):",
        "  APIDown · HighErrorRate · HighPredictLatencyP95 · FlagRateSurge",
    ], size=13)
    card(s, Inches(6.9), Inches(1.25), Inches(5.8), Inches(5.7), "Drift & retraining", [
        "GET /drift — live Evidently report",
        "  recent requests vs the training baseline",
        "  data drift + prediction (score) drift",
        "Retraining trigger (concrete):",
        "  ≥30% of features drift, or prediction drift",
        "  recall < 0.55 on newly labelled data",
        "  quarterly floor regardless",
    ], size=13)

    # 7. Architecture
    s = content_slide(prs, "7.  Architecture & Data Flow")
    if (HERE / "architecture.png").exists():
        pic = s.shapes.add_picture(str(HERE / "architecture.png"), 0, Inches(1.15), height=Inches(6.1))
        pic.left = int((SW - pic.width) / 2)
    bullets(s, Inches(0.4), Inches(7.0), Inches(12.5), Inches(0.5), [
        "Offline plane (dvc repro → MLflow → export) feeds artifacts/; online plane serves them. "
        "Monitoring & governance read the audit log and the trained model.",
    ], size=11)

    # 8. Demo
    s = content_slide(prs, "8.  Live Demo")
    bullets(s, Inches(0.6), Inches(1.4), Inches(12.1), Inches(5.6), [
        "/            demo UI — enter a patient, get risk + flag + SHAP factors",
        "/predict     JSON API — calibrated risk, flag, top contributing factors",
        "/governance  tabs: Model Card · Model Evaluation · Fairness Audit · SHAP",
        "/drift       live Evidently data + prediction drift report",
        "Grafana  (:3000)   service & model dashboards",
        "Prometheus (:9090)  metrics & firing alerts",
        "",
        "One command runs the whole stack:  docker compose up --build",
    ], size=18)

    out = HERE / "diabetes_readmission_demo.pptx"
    prs.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
