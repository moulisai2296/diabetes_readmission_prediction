# Diabetes Readmission Prediction — Design Notes & Learnings

A running journal of design concepts encountered while building this project.
Each entry captures *why* a choice was made so the principle is reusable elsewhere.

Format for each entry:
- **Context** — where in this project it came up
- **Principle** — the one-line takeaway
- **Explanation / analogy** — the mental model
- **Advantages / Tradeoffs**
- **Rule of thumb** — the quick test to apply it elsewhere

---

## Index

1. [DVC — versioning data like code](#1-dvc--versioning-data-like-code)
2. [MLflow — a lab notebook for experiments](#2-mlflow--a-lab-notebook-for-experiments)
3. [Grouped splitting — leakage you can't see in the metrics](#3-grouped-splitting--leakage-you-cant-see-in-the-metrics)
4. [Probability calibration — scores that mean what they say](#4-probability-calibration--scores-that-mean-what-they-say)
5. [Training/serving skew — one transform, two callers](#5-trainingserving-skew--one-transform-two-callers)
6. [Drift detection — the world moved, the model didn't](#6-drift-detection--the-world-moved-the-model-didnt)

---

## 1. DVC — versioning data like code

**Context:** Stage 1 of this project. Git must reproduce everything, but
`data_folder/diabetic_data.csv` is 19 MB of patient data that doesn't belong in a git
repo. We track it with DVC: `data_folder/diabetic_data.csv.dvc` is committed, the CSV
itself is not, and the cleaning step is a pipeline stage in `dvc.yaml`.

**Principle:** *Git versions the recipe; DVC versions the ingredients — via small
pointer files git CAN hold, standing in for big files it shouldn't.*

### Explanation / analogy

Think of a **coat check at a theatre**. You don't carry your heavy coat (the dataset)
to your seat (the git repo) — you hand it over and get a small numbered ticket. The
ticket is light, fits in your pocket, and uniquely identifies *exactly your coat*.

That ticket is the `.dvc` file. Look inside `data_folder/diabetic_data.csv.dvc` — it's
~5 lines of YAML containing an **MD5 hash** of the file's content, its size, and its
path. Git commits the ticket; the coat goes to the **cache** (`.dvc/cache/`), a
warehouse where every coat is shelved *by its hash* (content-addressed storage). Same
content → same hash → stored once, no matter how many tickets reference it.

Three mechanics on top of that:

1. **Pointer files** (`dvc add <file>`) — creates the ticket, moves the file to the
   cache, links it back into place, and adds the real file to `.gitignore` so you
   can't accidentally commit it.
2. **Remotes** (`dvc push` / `dvc pull`) — the cache can sync to shared storage (S3,
   GCS, a network drive). A teammate clones the git repo (gets tickets), runs
   `dvc pull` (exchanges tickets for coats). Without a remote, data lives only in the
   local cache — fine for solo work, the repo documents *which* data regardless.
3. **Pipelines** (`dvc.yaml` + `dvc repro`) — this is DVC acting like `make`. Our
   `clean` stage declares *command, dependencies, outputs*:
   - deps: `src/data/clean.py`, the two raw CSVs
   - outs: `data_folder/processed/cleaned.parquet`

   `dvc repro` hashes the deps; if nothing changed it skips the stage, if anything
   changed it re-runs and records the new hashes in `dvc.lock`. Committing
   `dvc.lock` means git history pins **code version + data version + output version
   together** — checkout any old commit and you know precisely which data produced
   which result. That is data lineage, and it's what "version your data & cleaning
   steps" actually means.

### Commands we use in this project

| Command | What it does |
|---|---|
| `uv run dvc init` | one-time setup; creates `.dvc/` (config + cache) |
| `uv run dvc add data_folder/diabetic_data.csv` | start tracking a data file (ticket + cache + gitignore) |
| `uv run dvc repro` | re-run pipeline stages in `dvc.yaml` whose deps changed; update `dvc.lock` |
| `uv run dvc status` | which stages/files are out of date vs `dvc.lock` |
| `uv run dvc dag` | visualize the stage dependency graph |
| `uv run dvc remote add -d storage <url>` | configure shared storage (when we need one) |
| `uv run dvc push` / `dvc pull` | sync cache with the remote (upload / download data) |
| `uv run dvc checkout` | restore data files to match the `.dvc`/`dvc.lock` of the current git commit |

The rhythm: **git checkout switches the tickets, dvc checkout/pull swaps the coats to
match.**

### Advantages

1. Git repo stays small and fast — reviews and clones never haul megabytes of data.
2. Exact reproducibility — every commit pins the precise data version it was built on.
3. `make`-style caching — `dvc repro` skips work whose inputs didn't change.
4. Deduplication — content-addressing stores identical data once.
5. Same workflow scales from a laptop to S3/GCS without changing the repo.

### Tradeoffs

- A second tool to learn; teammates must know `dvc pull` exists or they'll see only
  pointer files and wonder where the data went.
- If you edit a tracked file and forget `dvc add` (or bypass `dvc repro`), the ticket
  silently no longer matches the coat — `dvc status` is the guard.
- Sharing data needs a remote (extra infra); git hosting alone isn't enough.
- On Windows, cache linking falls back to copying (slower, doubles disk for big files).

### Rule of thumb

> If a file is too big or too binary for git, but your results depend on its exact
> version, commit a ticket (`.dvc` pointer), not the file — and if the file is
> *produced* by code, make it a `dvc.yaml` stage output instead, so the pipeline,
> not a human, keeps it fresh.

## 2. MLflow — a lab notebook for experiments

**Context:** Stage 3. We train a logistic baseline plus tuned XGBoost and LightGBM,
and must "track every experiment in MLflow" and later serve one registered winner.
`src/models/train.py` logs every run to a local SQLite-backed MLflow store
(`mlflow.db`, gitignored).

**Principle:** *If an experiment isn't recorded with its params, code and data
versions, it never happened — you can't compare, reproduce, or defend it.*

### Explanation / analogy

A chemist doesn't memorize "attempt #37 was the good one" — every attempt goes into a
**lab notebook**: ingredients (hyperparameters), procedure (code version), conditions
(data version), and measurements (metrics). Reviewers read the notebook, not the
chemist's memory.

MLflow is that notebook, with four parts:

1. **Tracking** — each training attempt is a *run*; we log params (model type, tuned
   hyperparameters), metrics (val PR-AUC, recall@precision, Brier), and artifacts
   (the fitted model itself). Runs group into an *experiment*
   (`diabetes-readmission`). `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`
   gives a sortable table — the baseline-vs-advanced comparison the brief grades.
2. **Backend store** — where runs/metrics live. MLflow ≥3.13 retired the `./mlruns`
   file store (we hit this: it raised "maintenance mode" and refused); a SQLite file
   is the lightweight replacement. Artifacts (model binaries) still go to a folder.
3. **Models** — a saved model with its signature (input schema) and environment, so
   serving loads exactly what training produced.
4. **Registry** — named, versioned models ("readmission-risk v3") with stage labels;
   this is what deployment and the rollback plan point at later.

DVC and MLflow are complementary notebooks: DVC remembers *data* versions along the
pipeline; MLflow remembers *attempts* and their outcomes.

### Advantages

1. Comparisons are honest — same metrics, computed by the same code, side by side.
2. Reproducibility — a run pins hyperparameters + code + model binary together.
3. The registry gives deployment a stable name to pull, decoupled from file paths.
4. Zero infra here: one SQLite file + an artifacts folder, both gitignored.

### Tradeoffs

- Local store = single machine; a team needs a shared tracking server (infra + auth).
- Logging discipline is on you — anything not logged (the random seed you "just
  tried") is lost; autolog helps but logs noise too.
- The SQLite store doesn't scale to thousands of concurrent runs (fine here).

### Rule of thumb

> The moment you train a *second* model variant, stop comparing from memory or
> terminal scrollback — log both to a tracker and let the table decide.

## 3. Grouped splitting — leakage you can't see in the metrics

**Context:** Stage 3, `src/models/split.py`. 23.5% of patients have multiple hospital
encounters — 46.2% of all rows. We split train/val/test by `patient_nbr`
(GroupShuffleSplit), and tune with `StratifiedGroupKFold`, so every patient lives in
exactly one partition.

**Principle:** *Split on the unit you must generalize to — here, the patient —
not on the row.*

### Explanation / analogy

Imagine studying for an exam with a stack of practice questions, where half the
questions appear **twice in the stack**. If you randomly deal the stack into
"practice" and "mock exam" piles, many mock-exam questions are ones you already saw
in practice. You'll ace the mock exam — and learn nothing about the real one.

Rows of the same patient are near-duplicate questions: same demographics, same
chronic conditions, often the same medications. A model can score well on a
row-split validation set simply by *recognizing the patient*, not by learning what
makes anyone readmission-prone. The trap: **metrics look great and nothing visibly
fails** — the lie only surfaces in production, where every patient is new.

Mechanics in this project:
- `GroupShuffleSplit` deals out *patients* (70/15/15), and each patient's rows follow
  them into their split. An assert + 4 tests enforce zero overlap.
- Inside training, hyperparameter search uses `StratifiedGroupKFold`: grouped (no
  patient straddles CV folds) *and* stratified (each fold keeps ~11.4% positives —
  important when the positive class is rare).
- The test split is produced once, here, and no training code path reads it.

### Advantages

1. Validation metrics estimate performance on *unseen patients* — the deployment
   reality at discharge time.
2. Honest model comparison — leakage would flatter the more memorization-capable
   model (the boosted trees) and bias the baseline-vs-advanced verdict.

### Tradeoffs

- Slightly unbalanced split sizes (patients carry different row counts) — ours landed
  within 0.2% of 70/15/15 anyway.
- Stratification + grouping together is approximate; target rates across splits vary
  a little (11.3–11.6% here — acceptable).

### Rule of thumb

> Before any split, ask: "what entity must this model work on that it has never seen
> before?" Patients, users, devices, documents — if any entity owns multiple rows,
> split by the entity. If you're unsure whether grouping matters, count rows per
> entity; any answer above 1 means it does.

## 4. Probability calibration — scores that mean what they say

**Context:** Stage 3, `src/models/finalize.py`. Our class-weighted XGBoost ranked
patients well (PR-AUC 0.239) but its "probabilities" were inflated: Brier score 0.206,
*worse than just always predicting the 11.4% base rate* (Brier 0.101). Isotonic
calibration dropped Brier to 0.097 without touching the ranking.

**Principle:** *A risk score shown to a human is a promise — "30%" must mean roughly
3 in 10. Class weighting breaks that promise; calibration restores it.*

### Explanation / analogy

A bathroom scale that always reads **5 kg heavy** is still perfectly good for telling
which of two people is heavier (ranking intact) — but you wouldn't quote its number as
your weight. Class weighting (`scale_pos_weight ≈ 7.8`) does exactly this to a
classifier: to fight imbalance it makes positives count ~8× in the loss, so the model
behaves as if readmission were ~8× more common — every probability reads "heavy".

Calibration re-labels the scale's dial. **Isotonic regression** learns a monotonic
mapping from raw score → honest probability on held-out folds: among patients scored
~0.6, what fraction *actually* readmitted? That fraction becomes the new output.
Monotonic means order is preserved — ROC-AUC and PR-AUC are unchanged by construction;
only the dial's numbers move.

Mechanics here: `CalibratedClassifierCV(method="isotonic", cv=<patient-grouped folds>)`
— fit the model on 2/3 of train, learn the mapping on the held-out 1/3, rotate, then
average. Grouped folds again: a patient straddling the model-fold and the
calibration-fold would leak. (`CalibratedClassifierCV` has no `groups=` argument — we
pass precomputed `StratifiedGroupKFold` splits as `cv`.)

Why it matters doubly here: our **threshold is cost-based** — "flag when
p × 0.20 × $15,000 > $300" only computes correctly if p is an honest probability.
With inflated scores the formula would flag nearly everyone.

**Measuring it:** Brier score = mean (predicted − actual)² — lower is better, and the
"always predict base rate" score (≈ prevalence × (1−prevalence) = 0.101 here) is the
bar to beat. A calibration curve (predicted bucket vs observed frequency) shows *where*
the dial lies; ours is logged as an MLflow artifact.

### Advantages

1. Scores are interpretable as real risk — essential when humans act on them.
2. Cost/utility-based thresholds become mathematically valid.
3. Free lunch for ranking: monotonic remapping leaves AUC metrics untouched.

### Tradeoffs

- Needs held-out data (CV folds) — slightly less data per model fit.
- Isotonic can overfit small calibration sets (<~1000 samples; we have 23k per fold).
  Platt scaling (sigmoid) is the low-data alternative.
- An ensemble of 3 calibrated models = 3× inference cost (negligible here).

### Rule of thumb

> If you reweighted, resampled, or otherwise fought class imbalance, your
> probabilities are lying. Check Brier against the "always predict the base rate"
> bar; if you lose, calibrate before anyone reads your scores as percentages.

## 5. Training/serving skew — one transform, two callers

**Context:** Stage 4. Training built features with `build_features.py` over a 100k-row
table; the live API scores one patient at a time. If the two paths compute features
even slightly differently, the model receives inputs unlike anything it trained on and
returns confident nonsense — with no error anywhere. We share `engineer()` between both
and persist the fitted vocabulary in `feature_spec.json`.

**Principle:** *The features at serving time must be produced by the same code, with
the same fitted parameters, as the features at training time — anything else is a
silent accuracy leak.*

### Explanation / analogy

A tailor measures you and sews a suit (training: fit the model to features). Months
later you order a second suit by phone (serving). If the assistant taking your phone
measurements rounds to the nearest inch while the tailor used centimetres, the suit
arrives subtly wrong — and nobody notices until you put it on. The fix isn't "measure
carefully," it's "use the *same ruler*."

Two kinds of skew bit at us, and each needed a different fix:

1. **Logic skew** — the engineered formulas (`total_prior_visits`, ICD-9 buckets,
   `age_ordinal`). Fix: there is exactly *one* function, `engineer()`, imported by both
   the training pipeline and the API's `featurize()`. Not "re-implemented identically"
   — literally the same function. The only safe amount of duplicated transform code is
   zero.

2. **State skew** — transforms that *learned something* from training data. Our
   rare-category lumping keeps medical specialties that were ≥1% *of the training set*;
   a single request has no frequencies to recompute from. And XGBoost's native
   categorical encoding maps each category to an integer **code by position** — if
   serving lists categories in a different order, "Cardiology" silently becomes a
   different number than it was in training. Fix: freeze that learned state at export
   time into `feature_spec.json` (the surviving vocabulary + the exact category order)
   and replay it at serving. The model artifact and the feature spec are exported
   together, as a matched pair.

The general lesson: any transform with a `.fit()` (scalers, encoders, imputers,
vocabularies) carries state that must travel from training to serving. Stateless
transforms only need shared code; stateful ones need shared code *and* shared
parameters.

### Advantages

1. A whole class of "great offline, broken online" bugs becomes structurally
   impossible — the ruler is identical by construction.
2. The serving contract is honest: callers send raw-ish facts, the server owns every
   derived feature, so a client can't drift the definition of `total_prior_visits`.
3. Export bundles model + spec, so deployment and rollback move one matched unit.

### Tradeoffs

- The training transform must be written to run on one row, not just a batch — no
  reliance on `df`-wide statistics inside the row-level path.
- Two artifacts must stay in sync; exporting them in one step (not by hand) is what
  prevents a stale spec against a fresh model.
- Sharing code couples the serving image to the training package (our API imports
  `src.features`) — fine here, but large shops sometimes extract a small shared
  transform library to avoid shipping all of training into the serving container.

### Rule of thumb

> Ask of every feature: "could the client compute this differently than training did?"
> If yes, move it server-side behind shared code. And for any transform that *learned*
> from data, export its fitted parameters alongside the model — never recompute them
> from a single request.

## 6. Drift detection — the world moved, the model didn't

**Context:** Stage 5, `src/monitoring/drift.py` + `GET /drift`. A model is frozen at
training time, but patients keep changing. We compare the recent live requests (rebuilt
from the prediction audit log) against a sample of the *training* distribution
(`artifacts/drift_reference.parquet`) using Evidently, and surface data + prediction
drift as a live report.

**Principle:** *A model assumes tomorrow looks like its training data; drift detection
is the smoke alarm that tells you that assumption expired — before the labels do.*

### Explanation / analogy

A trained model is a snapshot of the world *as it was*. The danger isn't that it
crashes when the world changes — it's that it keeps returning confident numbers while
quietly becoming wrong. Accuracy rots silently.

Think of a **thermostat you calibrated for your old house**. Move it to a drafty cabin
and it still reads a number and still clicks the heater on — it just clicks at the wrong
times, and you won't notice until you're cold. Drift detection is periodically holding
the cabin's actual temperature readings next to the original calibration chart and
asking: *"are these even the same shape anymore?"* You're not testing whether the
thermostat is broken (it works fine) — you're testing whether the world it was tuned for
still exists.

Two things drift, and the distinction matters:

1. **Data drift** — the *inputs* shift. An older population, a new EHR that codes
   "emergency" differently — feature distributions move away from training. Measured
   per column: K–S test for numeric features, chi-squared for categoricals.
2. **Prediction drift** — the *outputs* shift. The risk-score distribution moves (say
   40% score high-risk instead of the usual ~11%). Often a *symptom* of data drift, but
   also catches upstream pipeline bugs. We expose it by carrying the model's score as an
   extra numeric column in the same report.

Mechanics here: the **reference** is the train split scored once at export time; the
**current** window is rebuilt from `logs/predictions.jsonl` by re-running the same
`featurize()` the model is served with (so both sides share one feature definition — see
[note 5](#5-trainingserving-skew--one-transform-two-callers)). `/drift` refuses below 30
logged requests because the statistical tests are noise on tiny samples.

### Advantages

1. Early warning — you learn the input world changed *now*, not 30 days later when the
   readmission labels finally arrive.
2. Localizes the problem — the per-feature report says *which* columns moved, pointing
   straight at the likely cause (a changed feed, a new population).
3. Reuses what you already have — the audit log doubles as the drift window and the
   lineage trail; no separate data capture.

### Tradeoffs

- Drift ≠ decay. It tells you the inputs changed, *not* that the model got worse — the
  two usually correlate but not always. It's a trigger to investigate, not proof.
- Sensitive to window size and noisy on small samples; needs a sensible reference and a
  minimum sample floor.
- "How much drift is too much?" is a judgement call — a threshold (we use ≥30% of
  features, or prediction drift) that you'll tune against false-alarm tolerance.

### Rule of thumb

> Any model serving live traffic needs a reference distribution and a periodic "same
> shape?" check against it. Pick the reference as *exactly what the model learned from*
> (the train split, not val/test, not "all data"), and treat a drift alarm as "go look,"
> never as "the model is broken" — confirm against real outcomes before retraining.
