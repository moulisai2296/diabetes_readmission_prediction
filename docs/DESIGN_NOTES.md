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
