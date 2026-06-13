# Observability & Monitoring (Stage 5)

How the running service is watched: service health, model behaviour, data/prediction
drift, alerting, and the concrete retraining trigger.

Bring the stack up with `docker compose up -d`, then:

| What | Where |
|---|---|
| API + demo UI | http://localhost:8000/ |
| Live drift report | http://localhost:8000/drift |
| Raw metrics | http://localhost:8000/metrics |
| Prometheus (targets, alerts) | http://localhost:9090 |
| Grafana dashboard | http://localhost:3000 (admin/admin) |

## 1. Service metrics (Prometheus + Grafana)

The API exposes Prometheus metrics at `/metrics`, scraped every 15s:

| Metric | Meaning |
|---|---|
| `predict_requests_total` | request count (→ request rate) |
| `predict_errors_total` | failed predictions (→ error rate) |
| `predict_latency_seconds` | latency histogram (→ p95) |
| `predict_flagged_total` | patients flagged for follow-up |
| `predict_risk_score` | predicted-risk histogram (score distribution) |

The Grafana dashboard **Readmission Risk API** charts request rate, p95 latency, flag
rate, and mean predicted risk.

## 2. Model metrics (prediction audit log)

Every scored request is appended to `logs/predictions.jsonl` (one JSON object per line):
timestamp, model version, risk, flag, and the full input encounter. This is both the
**lineage/audit trail** (Stage 6) and the **window the drift report reads**. The log is
bind-mounted (`./logs`) so it survives container restarts.

The `predict_risk_score` histogram tracks the score *distribution over time* in Prometheus.

## 3. Drift (Evidently, live at `/drift`)

`GET /drift` rebuilds the served feature matrix from the recent audit-log window
(`DRIFT_WINDOW`, default 500) and compares it to the **training baseline**
(`artifacts/drift_reference.parquet` — a 5 000-row sample of the *train* split plus the
model's score on it), then returns Evidently's report:

- **Data drift** — per-feature distribution shift of the inputs vs training.
- **Prediction drift** — shift in the predicted-risk distribution (the score is carried
  as an extra numeric column).

It refuses below `MIN_DRIFT_SAMPLES` (default 30) logged predictions.

## 4. Alerting (Prometheus rules)

`alerts.yml` (loaded via `rule_files` in `prometheus.yml`, visible at
http://localhost:9090/alerts):

| Alert | Condition | Severity |
|---|---|---|
| `APIDown` | target unscrapeable 1m | critical |
| `HighErrorRate` | >5% of `/predict` errored over 5m | critical |
| `HighPredictLatencyP95` | p95 latency >500ms over 5m | warning |
| `FlagRateSurge` | >60% of patients flagged over 30m | warning |

Routing to a human (email/Slack/PagerDuty) is an Alertmanager concern, intentionally left
out of the local demo stack — the rules fire and are visible in Prometheus.

## 5. Retraining trigger (concrete)

Retrain when **any** of the following holds:

1. **Data drift** — Evidently reports drift in **≥ 30% of features**, *or* the prediction
   (risk-score) column itself drifts, on a rolling weekly window vs the training baseline.
2. **Performance decay** — once outcomes are known (30 days post-discharge), **recall at
   the operating threshold drops below 0.55** on the newly labelled month (vs ~0.6 at
   ship). Recall is the metric that matters: a missed readmission is the costly error.
3. **Calendar floor** — at least **quarterly**, regardless of the above, to absorb slow
   coding/population changes.

Retraining reruns the Stage 1–3 pipeline on data through the new cut-off, compares the
candidate to the incumbent on a held-out recent slice (PR-AUC + recall@threshold + a
Fairlearn re-check), and only promotes it in the MLflow registry if it wins. The previous
model version stays registered for one-command rollback.
