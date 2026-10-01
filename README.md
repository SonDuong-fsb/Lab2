# Lab 2 — ML Pipeline & Experiment Tracking

DDM501 · AI in DevOps, DataOps, MLOps · FSB

Lab 1 produced one model from one script. That does not survive contact with a
real team: nobody can say which data produced it, which hyperparameters won, or
why this version rather than the last one. This lab turns that script into a
pipeline whose every run is recorded, comparable, and — when it earns it —
promoted automatically.

Same credit default problem as Lab 1. Same data. The question changes from
*does it serve?* to *can you reproduce it, compare it, and decide about it?*

**This is the starter repository.** The skeleton, the data, `config.py`,
`data_ingestion.py` and `run_pipeline.py` are done. The stages that make it a
pipeline are marked `TODO`.

---

## What you have to complete

| File | TODOs | What it is |
|---|---|---|
| `pipeline/validation.py` | 4 | The three-level data quality gate |
| `pipeline/preprocessing.py` | 2 | Derived features and the ColumnTransformer |
| `pipeline/training.py` | 1 | MLflow tracking around the fit |
| `pipeline/evaluation.py` | 3 | Metrics, per-group metrics, fairness gap |
| `pipeline/registry.py` | 5 | Best run, register, alias, quality gate, promotion |
| `dags/credit_training_dag.py` | 6 | Five task bodies and the dependency graph |
| `tests/test_pipeline.py` | 5 classes | Everything except `TestDataIngestion` |
| `docker/airflow.Dockerfile` | 3 | User, constrained install, PYTHONPATH |
| `docker-compose.yml` | 1 | The MLflow service |

Read these before you start — they are the worked examples:
`pipeline/config.py`, `pipeline/data_ingestion.py`, `pipeline/run_pipeline.py`,
the `ingest` and `cleanup` tasks in the DAG, `beats_champion` and the helper
functions in `registry.py`, and `TestDataIngestion` in the test file.

**You are done when `.github/workflows/smoke.yml` passes.** It runs the pipeline,
asserts a model reached the `@champion` alias, runs the sweep, then installs
Airflow and parses the DAG. That workflow is the specification; this README is
the explanation.

---

## The pipeline

![The pipeline: five stages, and the three that decide whether a run produces anything](docs/lab2-pipeline.svg)

Each stage is a module in `pipeline/`. The CLI, the tests and the Airflow DAG
all call the same functions — the DAG orchestrates, it never reimplements.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Runs against a local file store at ./mlruns — no server needed
python -m pipeline.run_pipeline

# Look at what it recorded
mlflow ui --backend-store-uri ./mlruns      # http://localhost:5000
```

With the tracking server instead:

```bash
docker compose up -d mlflow
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/setup_mlflow.py             # confirms it is reachable
python -m pipeline.run_pipeline
```

Sweep the hyperparameter grid and print a leaderboard:

```bash
python -m experiments.run_experiments
python -m experiments.run_experiments --leaderboard-only --top 5
```

---

## The full stack

```bash
docker compose up -d --build

# MLflow   http://localhost:5000
# Airflow  http://localhost:8080   (airflow / airflow)
```

Unpause `credit_default_training` in the Airflow UI and trigger it. Seven tasks:

| Task | Does |
|---|---|
| `ingest` | Load the CSV, split it, write the split to the shared volume |
| `validate` | Schema, statistics and semantics. Raises and stops the DAG on failure |
| `train` | Fit the pipeline inside an MLflow run |
| `evaluate` | Aggregate metrics, per-group metrics, fairness gap — all logged |
| `decide` | Branch on the quality gate |
| `promote_model` / `skip_promotion` | Register and alias, or do nothing |
| `cleanup` | Remove the run directory down whichever branch ran |

---

## Two dependency sets, on purpose

`requirements.txt` has no Airflow in it. That is not an oversight.

Airflow pins several hundred transitive dependencies. Installing it next to
MLflow in one environment makes pip backtrack for minutes and often fails; when
it does succeed it silently downgrades things MLflow needs. So:

| File | Used by | Notes |
|---|---|---|
| `requirements.txt` | Your laptop, CI, the tests | numpy 2.2, pandas 2.2, scikit-learn 1.6 |
| `requirements-airflow.txt` | `docker/airflow.Dockerfile` only | numpy 1.24, pandas 2.1 — **pinned by Airflow's own constraints** |

The two sets disagree on numpy and pandas versions, and that is fine: they never
share an interpreter. Trying to reconcile them is how people lose an afternoon.

The Airflow image installs those packages **with Airflow's constraints file**.
Skipping the constraint is the fastest way to break a working scheduler.

---

## MLflow aliases, not stages

MLflow deprecated model registry stages (`Staging`, `Production`) in 2.9 and
will remove them. This lab uses **aliases**:

```python
client.set_registered_model_alias(name, alias="champion", version="4")
model = mlflow.sklearn.load_model("models:/credit-default-classifier@champion")
```

| | Stages (deprecated) | Aliases |
|---|---|---|
| Promotion | mutates the version's state | moves a pointer |
| History | overwritten | intact — versions are immutable |
| Count | four fixed names | as many as you need |

Two aliases are used here: `@champion` is what a serving layer would load,
`@challenger` is a candidate that passed the gate but did not beat the champion.

---

## The quality gate

`promote_model` never promotes on accuracy alone. Three checks, all must pass:

```
roc_auc    >= 0.70
pr_auc     >= 0.45
fairness_gap <= 0.10
```

and then the candidate must beat the current champion by a margin of 0.002 —
without the margin, noise triggers deployments.

The fairness check is the interesting one. `fairness_gap` is the largest
difference in *selection rate* (the share of applicants sent to review or
decline) between demographic groups. A model can be the most accurate candidate
in the sweep and still be refused promotion here.

Run the sweep and look at the leaderboard: logistic regression usually posts the
best ROC AUC **and** the widest fairness gap. Deciding what to do about that is
the point of the exercise, and Session 6 gives you the vocabulary for it.

---

## Project structure

```
.
├── pipeline/
│   ├── config.py           Everything configurable, all env-overridable
│   ├── data_ingestion.py   Load, stratified split, dataset stats
│   ├── validation.py       Schema / statistics / semantics gate
│   ├── preprocessing.py    Derived features + the sklearn ColumnTransformer
│   ├── training.py         Fit inside an MLflow run
│   ├── evaluation.py       Metrics, per-group metrics, fairness gap
│   ├── registry.py         Register, alias, quality gate, promotion
│   └── run_pipeline.py     CLI entry point
├── experiments/
│   └── run_experiments.py  Grid sweep + leaderboard
├── dags/
│   └── credit_training_dag.py
├── docker/
│   └── airflow.Dockerfile
├── data/credit_default.csv Committed — the lab needs no network
├── tests/test_pipeline.py
├── docker-compose.yml
├── requirements.txt
└── requirements-airflow.txt
```

---

## Tests

```bash
pytest tests/ -v --cov=pipeline --cov-report=term-missing
```

| Class | Asserts |
|---|---|
| `TestDataIngestion` | The split is stratified, reproducible, and leaks nothing |
| `TestValidation` | Every category of bad data is caught and stops the run |
| `TestPreprocessing` | Derived features are correct; the transformer lives inside the Pipeline |
| `TestTraining` | Every model type builds and fits; params override defaults |
| `TestEvaluation` | Metrics are in range and internally consistent; slices cover everyone |
| `TestQualityGate` | An accurate but unfair model is rejected; a missing metric fails closed |

`test_split_is_reproducible` looks trivial and is not. Without a fixed seed,
every metric comparison in MLflow measures split noise as much as model
quality — and you would never know, because the numbers still look plausible.

---

## Troubleshooting

**`Experiment 'credit-default-risk' not found`**
Nothing has been logged yet. Run the pipeline once, or `python scripts/setup_mlflow.py`.

**`Cannot reach the tracking server`**
`MLFLOW_TRACKING_URI` points at a server that is not running. Either
`docker compose up -d mlflow`, or unset the variable to fall back to `./mlruns`.

**Airflow UI shows the DAG as broken, with an import error**
The image does not have the pipeline's dependencies. Rebuild it:
`docker compose build airflow-scheduler`. Do not `pip install` inside a running
container — the change disappears on the next restart.

**`pip install` takes forever or fails after you added Airflow to requirements.txt**
Take it back out. See "Two dependency sets" above.

**`transition_model_version_stage is deprecated`**
You are using the old stage API. Use `set_registered_model_alias` instead.

**Every test passes before you have written anything**
A stub whose body is `pass` returns `None`, and pytest counts that as a pass.
Delete the `pass` as you implement each test.

**`airflow dags list` shows the DAG but the graph in the UI is a flat row**
You have not wired the dependencies yet — TODO 6 at the bottom of the DAG file.

**Every experiment run gets the same metrics**
Check that you are passing the sweep's parameters into `train_model`. It is easy
to build the config dict and never use it — the runs then differ only by name.

**The DAG runs but nothing appears in MLflow**
Inside the compose network the tracking server is `http://mlflow:5000`, not
`http://localhost:5000`. `localhost` inside a container is that container.
