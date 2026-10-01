# DDM501 Lab 2 — Report

All numbers below come from runs on the committed `data/credit_default.csv`
(30,000 rows, seed 501, 80/20 stratified split, decision threshold 0.30),
recorded in a local MLflow file store.

## 1. Pipeline design

| Stage | Module | Owns |
|---|---|---|
| Ingest | `data_ingestion.py` | Reading the CSV, the seeded stratified split, dataset statistics |
| Validate | `validation.py` | The three-level data quality gate and the report that travels with the run |
| Train | `preprocessing.py`, `training.py` | Derived features, the sklearn `Pipeline` (transformer + classifier), the MLflow run |
| Evaluate | `evaluation.py` | Aggregate metrics, per-group metrics, fairness gap, all logged to the training run |
| Promote | `registry.py` | Quality gate, registration, `@champion` / `@challenger` aliases |

**Why validation sits between ingestion and training.** A model trained on bad
data still trains, still scores well on a test split drawn from the same bad
data, and still registers. The failure is invisible downstream. Validation is
the only point where the pipeline can say "the data is wrong" instead of "the
model is weak", and it is cheap compared with a fit. Placing it before any
fitting also means a failed validation stops the DAG before a model exists. When
validation passes, its report is logged as `validation_report.json`, so the
model carries the evidence that it was trained on checked data.

All three validation levels return error lists instead of raising, so one run
reports every problem. The gate rejects a missing column, a non-numeric column,
too few rows, more than 2% missing per column, a target positive rate outside
[0.05, 0.60], and values outside the documented domains and ranges.

## 2. Experiment analysis

Sweep of 7 configurations, ranked by ROC AUC:

| run | model | roc_auc | pr_auc | recall | fairness gap |
|---|---|---|---|---|---|
| logreg-01 | logreg (C=0.1) | 0.7511 | 0.5534 | 0.5068 | 0.0558 |
| logreg-02 | logreg (C=1.0) | 0.7511 | 0.5534 | 0.5068 | 0.0544 |
| hgb-05 | hgb (200 it, lr 0.10, depth 4) | 0.7486 | 0.5468 | 0.4832 | 0.0368 |
| rf-03 | rf (200 trees, depth 8) | 0.7480 | 0.5391 | 0.4868 | 0.0332 |
| rf-04 | rf (300 trees, depth 12) | 0.7477 | 0.5402 | 0.4853 | 0.0341 |
| hgb-07 | hgb (500 it, lr 0.03, depth 8) | 0.7475 | 0.5435 | 0.4853 | 0.0276 |
| hgb-06 | hgb (300 it, lr 0.06, depth 6) | 0.7473 | 0.5444 | 0.4817 | 0.0306 |

What the sweep says:

1. **The models are nearly indistinguishable on accuracy.** The whole spread in
   ROC AUC is 0.0038. The three hgb runs sit within 0.0013 of each other, which
   is smaller than the 0.002 promotion margin. Hyperparameter tuning bought
   nothing measurable; the features (utilisation, delay counts) carry the signal,
   not the model family.
2. **The spread in fairness is much larger than the spread in accuracy.** The
   gap ranges from 0.0276 to 0.0558, roughly a factor of two, while AUC moves by
   half a percent. The choice of model is therefore mostly a choice about the
   gap.
3. **Logistic regression has the best AUC and recall and the widest gap.** It
   selects more applicants for review overall (recall 0.507 vs 0.482), and the
   extra selections fall unevenly across SEX.
4. **`C` does nothing for logistic regression here** (identical AUC and PR AUC at
   C=0.1 and C=1.0), because the features are scaled and the problem is
   well-conditioned. The two runs differ only in the fairness gap, by 0.0014.
5. **Limits of this evidence.** One split, one seed, no cross-validation, so
   differences below roughly 0.002 AUC should not be read as a ranking. The
   dataset is generated, and `scripts/make_dataset.py` deliberately builds in a
   SEX effect on the score, so the gap measured here reflects that design choice
   and says nothing about a real portfolio.

Every run passes the default gate (ROC AUC ≥ 0.70, PR AUC ≥ 0.45, gap ≤ 0.10),
so at the default thresholds the gate does not separate these candidates.

## 3. The promotion decision

**I would promote `hgb-05`** (ROC AUC 0.7486, gap 0.0368).

Against logistic regression it gives up 0.0025 AUC and 0.024 recall, and gets a
selection-rate gap about a third narrower (0.0368 vs 0.0558). The AUC loss is
about the size of the promotion margin, so it is within noise, while the
fairness difference is well outside it. Of the candidates with a gap under
0.04, `hgb-05` has the highest AUC. `hgb-07` has the narrowest gap (0.0276) but
costs another 0.001 AUC with no clear benefit over `hgb-05`.

**The gate has to be set to make that decision automatically.** The pipeline
promotes the highest-AUC candidate that clears the gate, and the default
`MAX_FAIRNESS_GAP` of 0.10 lets logistic regression through. To encode the
choice, set `MAX_FAIRNESS_GAP=0.04`. That rejects both logreg runs (0.0558,
0.0544) and leaves `hgb-05` as the best passing model. The threshold is a
policy decision, not a measured one: 0.04 is the value that matches my
preference on this sweep, and I did not test how it behaves on other data.

Caveat on the metric: the gap is a difference in selection rate (demographic
parity). It shows the model treats the groups differently, but it does not say
whether the difference is justified by genuinely different default rates, and I
did not analyse that. It should be read together with the per-group metrics in
each run's `evaluation.json`.

## 4. Orchestration

DAG `credit_default_training`:
`ingest → validate → train → evaluate → decide → [promote_model | skip_promotion] → cleanup`.

Every task body imports and calls the `pipeline` package; the DAG contains no
training logic of its own, so the scheduled model is the tested model.

| Travels through XCom (metadata) | Travels through the shared volume (data) |
|---|---|
| `run_dir` path, `mlflow_run_id` | `split.joblib` (train/test frames) |
| `data_stats` (4 scalars) | `raw.parquet` |
| `validation_report` (small dict of lists) | `model.joblib` |
| `metrics` (scalars only), `quality_gate`, `promotion` | `evaluation.json` (includes per-group metrics) |

XCom values are serialised into Airflow's metadata database and have a size
limit, so frames and models stay on disk and only paths and scalar results are
pushed. `evaluate` writes the nested per-group metrics to a file and pushes only
the flat scalar metrics.

`decide` is a `BranchPythonOperator` that returns the task id `promote_model` or
`skip_promotion`. `cleanup` uses `trigger_rule="none_failed_min_one_success"`
because a branch always skips one side; with the default `all_success`, cleanup
would be skipped on every run and leave the run directory behind. `validate`
raises on bad data, which fails the task and prevents `train` from ever running.

## 5. Reproducibility

Every training run logs what is needed to rebuild it:

| Logged | Where |
|---|---|
| Model type and hyperparameters | run params |
| `random_state`, `test_size` | run params |
| `data_sha256` (first 16 hex chars of the CSV hash), row counts, positive rate | run params |
| Validation report | `validation_report.json` |
| Exact feature columns the model expects | `feature_columns.json` |
| Fitted preprocessing + classifier | `model/` (one sklearn `Pipeline`) |
| Library versions | `model/requirements.txt` (for example scikit-learn 1.6.0, numpy 2.2.1, pandas 2.2.3) |
| Metrics, per-group metrics, fairness gap | run metrics and `evaluation.json` |

To reproduce `hgb-05`:

```bash
pip install -r requirements.txt
python -m pipeline.run_pipeline --model-type hgb --max-iter 200 --learning-rate 0.10 --max-depth 4 --no-register
```

and check that `data_sha256` matches the logged value. I confirmed the
mechanism on hgb-06: the CI run (`--model-type hgb`) and sweep run `hgb-06` use
the same configuration and produced identical results (ROC AUC 0.7473, PR AUC
0.5444, recall 0.4817, gap 0.0306).

Reproducibility depends on pinned library versions: a different scikit-learn
version can change results and can fail to load a model pickled by another
version.
