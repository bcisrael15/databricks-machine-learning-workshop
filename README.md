# Customer Churn — End-to-End MLflow on Databricks

A hands-on tutorial through the **full MLflow / MLOps lifecycle** on Databricks, using a
synthetic telecom customer-churn dataset. You start from raw data and finish with a
registered, served, and monitored model — touching feature engineering, experiment tracking,
hyperparameter tuning, model evaluation, the Unity Catalog Model Registry, real-time serving,
batch inference, and Lakehouse Monitoring along the way.

It's self-service: **download this folder as a ZIP → import it into your Databricks workspace
→ run the notebooks in order.**

### What you'll build

```
customers ─┐
           ├─→ feature table ─→ tuned LightGBM model ─→ real-time serving endpoint
service_tickets ┘                 (registered + aliased)  │
                                                           └─→ batch predictions ─→ monitoring
```

---

## Prerequisites

- A **Databricks workspace** with **Unity Catalog** and **serverless** compute (the notebooks
  pin the serverless `environment_version = "5"`).
- A **catalog you can create a schema in**. If it's a shared catalog, an admin grants it once:
  ```sql
  GRANT USE CATALOG ON CATALOG <catalog> TO `account users`;
  GRANT CREATE SCHEMA ON CATALOG <catalog> TO `account users`;
  ```
- **No `pip` or internet** required for the default data path; **no** SQL warehouse or
  Foundation Model API needed. (Only the optional `03b` AutoML notebook needs a classic ML
  Runtime cluster — see below.)

---

## Configuration — two widgets

Every notebook has two widgets at the top:

- **`catalog`** — the Unity Catalog catalog to build in (default `main`). Set this to yours.
- **`schema`** — your schema, pre-filled with a per-user name like `churn_workshop_<username>`.
  Keep the default so your work stays isolated.

The serving endpoint name (`churn-model-<username>`) is derived automatically. Widgets are
per-notebook, so glance at them each time you open a new notebook and confirm they point at
your catalog/schema before running.

---

## Setup

1. **Download this folder as a ZIP and import it** into your workspace (Workspace → **Import**).
   Import the **whole folder** so the structure is preserved — the notebooks load shared config
   via `%run ../_resources/00_config`, which only resolves if `_resources/` and `notebooks/`
   stay siblings.
2. Open **`_resources/01_setup_slim`**, set the `catalog` widget, leave `data_source` on
   `generate`, and **Run All**. It creates your schema and three tables in under a minute — no
   pip required.
   - *Locked-down / air-gapped workspace?* Set `data_source` to **`volume_csv`**, upload the
     bundled `data/*.csv` files to the `raw` Volume the setup creates, and re-run the load cell.

---

## The tutorial (run in order)

| Notebook | What you do | Key features |
|----------|-------------|--------------|
| `01_overview` | Explore the churn dataset and its signals | — |
| `02_feature_engineering` | Build a Unity Catalog **feature table** + an on-demand feature function | Feature Engineering client, feature tables, lineage |
| `03_train_and_register_model` | Tune a LightGBM model with Optuna, evaluate it, and register it | MLflow tracking, nested runs, `mlflow.evaluate`, model signatures, UC Model Registry, Champion/Challenger aliases, SHAP |
| `03b_automl_hyperparameter_tuning` *(optional)* | Solve the same problem with Databricks **AutoML** | `databricks.automl` — **requires a classic ML Runtime cluster** |
| `04_model_serving` | Deploy the Champion model to a real-time endpoint and call it | Model Serving, scale-to-zero, load-by-alias |
| `05_batch_inference` | Score a batch and write a predictions table | `mlflow` model loading, Spark inference |
| `06_monitoring` | Set up baseline, drift, and quality monitoring | Lakehouse Monitoring, drift tests, custom metrics, alerting |

Each notebook (except `01`) begins with a `%pip install` + Python restart — expect ~a minute
of setup at the top. If a cell ever fails with `ModuleNotFoundError` right after the install,
re-run that first cell (it restarts Python) and continue.

### A note on monitoring (`06`)

Creating a monitor returns quickly, but its **first refresh runs asynchronously** and takes
several minutes to populate the metric tables and the dashboard. If a metric cell shows 0 rows,
the refresh simply hasn't finished — re-run the cell once it completes. (Inference monitors
only profile data within a ~30-day window, so the notebook stamps its prediction windows
relative to *today*.)

### The optional AutoML notebook (`03b`)

`03b` reproduces the tuning step using Databricks AutoML. It needs a **classic ML Runtime
cluster** — AutoML does not run on serverless — and on DBR 18.0 ML+ it installs
`databricks-automl-runtime` (AutoML is no longer bundled in the runtime image there).

---

## Cleanup

When you're done, open **`_resources/99_cleanup`**, confirm the `catalog`/`schema` widgets
match what you used, type `DELETE` in the confirmation widget, and **Run All**. It removes
everything you created: your schema (`CASCADE`, so all tables, the feature table, and both
models), the serving endpoint, the monitor and its dashboard, and your MLflow experiments.

---

## Repository structure

```
├── _resources/
│   ├── 00_config.py          # catalog/schema widgets + derived names (loaded by every notebook)
│   ├── 01_setup_slim.py      # One-time setup: 3 tables (generate | volume_csv), < 1 min, no pip
│   ├── data_generators.py    # Synthetic data generators (dependency-free)
│   └── 99_cleanup.py         # Removes everything you created
├── data/                     # Bundled CSVs for the volume_csv route
│   ├── customers.csv
│   ├── service_tickets.csv
│   └── churn_labels.csv
└── notebooks/
    ├── 01_overview.py
    ├── 02_feature_engineering.py
    ├── 03_train_and_register_model.py
    ├── 03b_automl_hyperparameter_tuning.py
    ├── 04_model_serving.py
    ├── 05_batch_inference.py
    └── 06_monitoring.py
```

---

## Dataset

A self-contained synthetic telecom dataset generated at setup time (seed-fixed, ~33% churn):

| Table | Rows | Contents |
|-------|------|----------|
| `customers` | ~5,000 | Customer profiles, demographics, service mix, and the churn label |
| `service_tickets` | ~15,000 | Support tickets per customer |
| `churn_labels` | ~5,000 | Target variable with train / validation / test splits |

All data is synthetic — no real customers or PII.
