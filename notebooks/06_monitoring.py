# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Customer Churn — End-to-End MLflow
# MAGIC ## Notebook 06 — Lakehouse Monitoring
# MAGIC
# MAGIC
# MAGIC Build **production-grade model monitoring** using Databricks Lakehouse Monitoring. This capstone ties together everything you've built — scoring the model, detecting drift, tracking model quality, and setting up alerting.
# MAGIC
# MAGIC ### What you'll cover
# MAGIC | Concept | What it does |
# MAGIC |---------|-------------|
# MAGIC | **Baseline comparison** | Compare production predictions against training-distribution baseline |
# MAGIC | **Drift detection** | KS, chi-squared, PSI, Wasserstein, Jensen-Shannon, total variation |
# MAGIC | **Model quality** | Classification metrics (accuracy / precision / recall / F1) when labels are present |
# MAGIC | **Custom metrics** | Business KPIs via Jinja SQL templates |
# MAGIC | **Alerting** | SQL-based threshold alerts on drift metrics |
# MAGIC | **Dashboard** | Auto-generated Lakeview dashboard |

# COMMAND ----------

# MAGIC %md
# MAGIC > **How the refresh works.** Creating a monitor returns quickly (~30s to `ACTIVE`), but
# MAGIC > its **first refresh runs asynchronously** and takes several minutes to populate the
# MAGIC > profile/drift metric tables and the dashboard. The wait cell below polls for it; if a
# MAGIC > metric cell shows 0 rows, the refresh simply hasn't finished yet — re-run the cell once
# MAGIC > it completes.
# MAGIC >
# MAGIC > **Best practice — keep inference timestamps recent.** Inference monitors only profile
# MAGIC > data within a ~30-day lookback. The prediction windows below are stamped relative to
# MAGIC > *today* for exactly this reason: stale timestamps would leave the INPUT/drift metrics
# MAGIC > empty while only the baseline profile populates.

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.102.0 mlflow==3.8.1 scikit-learn==1.6.1 lightgbm==4.6.0 "numpy<2" -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Fast-fail env check. On serverless the %pip install above occasionally doesn't
# activate on the first attempt; catch that here with a clear message instead of a
# cryptic model-unpickle error later (loading the LightGBM pipeline needs lightgbm).
try:
    import lightgbm, mlflow
    print(f"✓ env ready — lightgbm {lightgbm.__version__}, mlflow {mlflow.__version__}")
except ModuleNotFoundError as e:
    raise RuntimeError(
        f"'{e.name}' is missing — the %pip install in the previous cell didn't take effect. "
        f"Re-run the first cell (it restarts Python), then continue."
    ) from e

# COMMAND ----------

import re
_u = spark.sql("SELECT current_user()").collect()[0][0]
_slug = re.sub(r"[^a-z0-9_]", "_", _u.split("@")[0].lower()).strip("_") or "user"
dbutils.widgets.text("catalog", "main", "1. Catalog (shared)")
dbutils.widgets.text("schema", f"churn_workshop_{_slug}", "2. Schema (yours)")

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
    MonitorInfoStatus,
    MonitorMetric,
    MonitorMetricType,
)
from pyspark.sql import functions as F
import mlflow
from mlflow.tracking import MlflowClient

w = WorkspaceClient()
mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

predictions_table = predictions_table_name
baseline_table = f"{catalog}.{schema}.churn_predictions_baseline"

# The monitor's model_id_col value must match the format the serving endpoint emits
# in its auto-captured inference table: "<fully-qualified model name> version <N>".
champion_version = mlflow_client.get_model_version_by_alias(model_name, "Champion").version
monitor_model_id = f"{model_name} version {champion_version}"
print(f"Monitoring model: {monitor_model_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Baseline Table
# MAGIC
# MAGIC A monitor needs a **baseline** — the distribution the model was trained on. If we compared the predictions table against itself, drift would always be zero.
# MAGIC
# MAGIC We score the **training split** with the Champion model. The resulting predictions + features + ground-truth labels become the reference distribution.

# COMMAND ----------

# Score training split — join labels to features, add the on-demand feature, predict.
train_labels = spark.table("churn_labels").filter("split = 'train'").select("customer_id")
features = spark.table(feature_table_name).drop("update_timestamp")
train_df = (
    train_labels
    .join(features, on="customer_id", how="inner")
    .withColumn(
        "avg_price_increase",
        F.expr(f"{catalog}.{schema}.avg_price_increase(monthly_charges, tenure_months)")
    )
)

model = mlflow.sklearn.load_model(f"models:/{model_name}@Champion")
feature_cols = [c for c in train_df.columns if c != "customer_id"]

train_pdf = train_df.toPandas()
# Cast prediction to float — monitoring requires prediction and label types to match (both DOUBLE).
train_pdf["prediction"] = model.predict(train_pdf[feature_cols]).astype("float64")
train_pdf["churn_probability"] = model.predict_proba(train_pdf[feature_cols])[:, 1]

# Ground truth for the training rows: Yes → 1.0, No → 0.0 (matches LabelEncoder used in nb 03).
churn_yes_no = (
    spark.table("churn_labels")
    .filter("split = 'train'")
    .select("customer_id", F.when(F.col("churn") == "Yes", 1.0).otherwise(0.0).alias("churn_label"))
    .toPandas()
)
train_pdf = train_pdf.merge(churn_yes_no, on="customer_id", how="left")

baseline_df = spark.createDataFrame(
    train_pdf[["customer_id"] + feature_cols + ["prediction", "churn_probability", "churn_label"]]
)

# Inference monitors only profile data within a ~30-day lookback, so the window
# timestamps MUST be recent — hardcoded calendar dates age out and the monitor profiles
# nothing (INPUT/drift stay empty while only the baseline shows). Stamp everything
# relative to TODAY, and use two distinct recent days (see daily granularity below).
from datetime import datetime, timedelta
_today = datetime.utcnow().date()
BASELINE_TS = str(_today - timedelta(days=25))
WINDOW1_TS  = str(_today - timedelta(days=14))
WINDOW2_TS  = str(_today - timedelta(days=2))
print(f"Baseline anchor: {BASELINE_TS} | windows: {WINDOW1_TS}, {WINDOW2_TS}")

# model_id matches the serving endpoint convention.
baseline_df = (
    baseline_df
    .withColumn("inference_timestamp", F.lit(BASELINE_TS).cast("timestamp"))
    .withColumn("model_id", F.lit(monitor_model_id))
)

baseline_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(baseline_table)
print(f"Baseline table created: {baseline_table} ({spark.table(baseline_table).count()} rows)")

# COMMAND ----------

# Baseline distribution sanity check — counts and feature averages per predicted class.
display(
    spark.table(baseline_table)
    .groupBy("prediction")
    .agg(
        F.count("*").alias("count"),
        F.round(F.avg("monthly_charges"), 2).alias("avg_monthly_charges"),
        F.round(F.avg("tenure_months"), 1).alias("avg_tenure"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build the Windowed Predictions Table
# MAGIC
# MAGIC The monitor reads from a **single source table** that already contains every (model_id, window) row it needs. We score the test split twice:
# MAGIC
# MAGIC The two windows are stamped with **recent dates** (relative to today) so they fall
# MAGIC inside the inference monitor's ~30-day profiling lookback:
# MAGIC
# MAGIC | Window | Features | Cause we're simulating |
# MAGIC |--------|----------|------------------------|
# MAGIC | ~14 days ago | original | normal production scoring |
# MAGIC | ~2 days ago | `monthly_charges` +15%, `total_charges` +20%, `tenure_months` +6, 70% forced to month-to-month contracts | pricing increase + ageing customer base + market shift |
# MAGIC
# MAGIC Both windows get scored by the **same Champion model** and labelled with the same `model_id`, so the monitor can group them and compute drift against the baseline.
# MAGIC
# MAGIC > We **rebuild the table in one `overwrite` write** (no ALTER COLUMN, no UPDATE, no MERGE). Inference Log monitors are sensitive to mid-stream schema churn — a single clean write produces a predictable Delta history that the monitor reliably profiles.

# COMMAND ----------

# Read test customers + features once — same join as notebook 05.
test_labels = (
    spark.table("churn_labels")
    .filter("split = 'test'")
    .select(
        "customer_id",
        F.when(F.col("churn") == "Yes", 1.0).otherwise(0.0).alias("churn_label"),
    )
)
inference_features = (
    test_labels
    .join(features, on="customer_id", how="inner")
    .withColumn(
        "avg_price_increase",
        F.expr(f"{catalog}.{schema}.avg_price_increase(monthly_charges, tenure_months)")
    )
)

base_pdf = inference_features.toPandas()
# Stable column ordering for both windows.
feature_only_cols = [c for c in inference_features.columns if c not in ("customer_id", "churn_label")]


def _score_window(pdf, inference_timestamp):
    """Score pdf with the Champion model and stamp it with the given window timestamp."""
    pdf = pdf.copy()
    pdf["prediction"] = model.predict(pdf[feature_only_cols]).astype("float64")
    pdf["churn_probability"] = model.predict_proba(pdf[feature_only_cols])[:, 1]
    pdf["inference_timestamp"] = inference_timestamp
    pdf["model_id"] = monitor_model_id
    return pdf


# Window 1 (recent) — original features, original scores.
window1 = _score_window(base_pdf, WINDOW1_TS)

# Window 2 (more recent) — drifted features, then re-score so predictions naturally shift.
import numpy as np
np.random.seed(42)
drifted_pdf = base_pdf.copy()
drifted_pdf["monthly_charges"] = (drifted_pdf["monthly_charges"] * 1.15).round(2)
drifted_pdf["total_charges"] = (drifted_pdf["total_charges"] * 1.20).round(2)
drifted_pdf["tenure_months"] = drifted_pdf["tenure_months"] + 6
mask = np.random.rand(len(drifted_pdf)) < 0.70
drifted_pdf.loc[mask, "contract_type"] = "Month-to-month"
# Recompute avg_price_increase since monthly_charges + tenure_months changed.
drifted_pdf["avg_price_increase"] = drifted_pdf["monthly_charges"] / drifted_pdf["tenure_months"].clip(lower=1)
window2 = _score_window(drifted_pdf, WINDOW2_TS)

# Stable column order across both windows; pandas concat preserves dtypes.
import pandas as pd
col_order = (
    ["customer_id"]
    + feature_only_cols
    + ["prediction", "churn_probability", "inference_timestamp", "model_id", "churn_label"]
)
windowed_pdf = pd.concat([window1[col_order], window2[col_order]], ignore_index=True)

windowed_df = (
    spark.createDataFrame(windowed_pdf)
    .withColumn("inference_timestamp", F.col("inference_timestamp").cast("timestamp"))
)

# Single clean overwrite — replaces whatever notebook 05 wrote with the monitor-ready version.
(
    windowed_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(predictions_table)
)
print(f"✓ Predictions table rebuilt with {spark.table(predictions_table).count()} rows across 2 windows")

# COMMAND ----------

# Window manifest — confirms both windows landed with the expected row count and label coverage.
display(spark.sql(f"""
    SELECT inference_timestamp,
           COUNT(*) AS rows,
           ROUND(100.0 * AVG(prediction), 1) AS pct_predicted_churn,
           SUM(CASE WHEN churn_label IS NOT NULL THEN 1 ELSE 0 END) AS labeled_rows
    FROM {predictions_table}
    GROUP BY inference_timestamp
    ORDER BY inference_timestamp
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the Lakehouse Monitor
# MAGIC
# MAGIC One API call — Databricks provisions the monitor, runs an initial refresh on a serverless job, writes per-column statistics to `*_profile_metrics`, drift tests to `*_drift_metrics`, and builds a Lakeview dashboard.
# MAGIC
# MAGIC We configure it as an **InferenceLog** monitor with:
# MAGIC - `prediction_col="prediction"` and `label_col="churn_label"` → drift **and** model quality
# MAGIC - `model_id_col="model_id"` → metrics are sliced per model version
# MAGIC - `timestamp_col="inference_timestamp"` and `granularities=["1 day"]` → one row per (column, window)
# MAGIC - A custom metric: `high_risk_percentage` — the % of churn=1 predictions per window

# COMMAND ----------

# Idempotency: delete any existing monitor before recreating. Without this the create
# call would fail on re-run with "monitor already exists".
try:
    w.quality_monitors.get(table_name=predictions_table)
    w.quality_monitors.delete(table_name=predictions_table)
    print(f"Deleted existing monitor on {predictions_table}")
    # Give the backend a moment to release the metric tables it owns.
    time.sleep(10)
except Exception:
    print("No existing monitor — creating fresh")

# COMMAND ----------

# Custom metric: % of high-risk (predicted churn=1) per time window. The {{prediction_col}}
# placeholder is replaced by the monitor with the configured prediction column name.
high_risk_metric = MonitorMetric(
    type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
    name="high_risk_percentage",
    input_columns=[":table"],
    definition="100.0 * SUM(CASE WHEN {{prediction_col}} = 1 THEN 1 ELSE 0 END) / COUNT(*)",
    output_data_type="DOUBLE",
)

username = spark.sql("SELECT current_user()").first()[0]

# Capture wall-clock just before create so we can identify "our" refresh in list_refreshes
# even if the SDK returns a stale or yet-to-be-persisted refresh_id.
t_create_ms = int(time.time() * 1000)

monitor_info = w.quality_monitors.create(
    table_name=predictions_table,
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="prediction",
        label_col="churn_label",
        model_id_col="model_id",
        timestamp_col="inference_timestamp",
        granularities=["1 day"],
    ),
    baseline_table_name=baseline_table,
    custom_metrics=[high_risk_metric],
    output_schema_name=f"{catalog}.{schema}",
    assets_dir=f"/Workspace/Users/{username}/databricks_lakehouse_monitoring/{predictions_table}",
)
print(f"Monitor create() returned status={monitor_info.status.value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wait for the Monitor to Go Active + Print Dashboard URL
# MAGIC
# MAGIC `create()` provisions the monitor and triggers an initial refresh asynchronously. The refresh runs on a serverless job that profiles the baseline + source, computes drift, evaluates classification metrics (because `label_col` is set) and the custom metric.
# MAGIC
# MAGIC We block only on the monitor going `ACTIVE` (~30s). The first refresh's INPUT / drift / quality rows are written **in the background** over the next 5–15 minutes — the cells below query them. If a cell returns 0 rows on first run, the refresh just hasn't materialised that section yet; re-execute the cell once the dashboard shows data.

# COMMAND ----------

profile_metrics_table = f"{catalog}.{schema}.churn_predictions_profile_metrics"
drift_metrics_table = f"{catalog}.{schema}.churn_predictions_drift_metrics"


def wait_for_monitor_active(table_name, timeout_s=300, poll_s=10):
    deadline = time.time() + timeout_s
    while True:
        info = w.quality_monitors.get(table_name=table_name)
        if info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE:
            print(f"  ✓ monitor ACTIVE")
            return info
        if info.status in (
            MonitorInfoStatus.MONITOR_STATUS_ERROR,
            MonitorInfoStatus.MONITOR_STATUS_FAILED,
        ):
            raise RuntimeError(f"Monitor reached terminal state {info.status.value}")
        if time.time() > deadline:
            raise TimeoutError(f"Monitor still {info.status.value} after {timeout_s}s")
        print(f"  monitor status={info.status.value} — waiting...")
        time.sleep(poll_s)


wait_for_monitor_active(predictions_table)

info = w.quality_monitors.get(table_name=predictions_table)
print(f"Assets folder: {info.assets_dir}")
print(f"Dashboard ID:  {info.dashboard_id}")
print(f"Dashboard URL: {w.config.host.rstrip('/')}/dashboardsv3/{info.dashboard_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Wait for the Metric Tables to Be Created
# MAGIC
# MAGIC The monitor's first refresh creates `*_profile_metrics` and `*_drift_metrics` itself — they don't exist until the refresh runs. We block for up to 15 minutes for the tables to **come into existence** (which is the minimum gate for the display cells below to read them). We don't require windowed INPUT / drift rows — those land at the same time on a healthy backend, but if the refresh is slow, the display cells simply return 0 rows and you can re-run them later.

# COMMAND ----------

def _table_counts():
    """Return (n_baseline_cols, n_input_windows, n_drift_windows). Each is 0 if the
    table or row class isn't present yet. Returns None if either table doesn't
    exist at all — the caller treats that as "keep polling."
    """
    try:
        n_baseline = spark.table(profile_metrics_table).filter("log_type = 'BASELINE'").select("column_name").distinct().count()
    except Exception:
        return None
    try:
        n_input = spark.table(profile_metrics_table).filter("log_type = 'INPUT' AND slice_key IS NULL AND window IS NOT NULL").select("window").distinct().count()
    except Exception:
        n_input = 0
    try:
        n_drift = spark.table(drift_metrics_table).filter("slice_key IS NULL AND window IS NOT NULL").select("window").distinct().count()
    except Exception:
        n_drift = 0
    return n_baseline, n_input, n_drift


def poll_for_metric_tables(timeout_s=1500, poll_s=30):
    """Block until BASELINE rows are present in profile_metrics — that's the stable
    signal that the refresh has finished creating both tables and writing at least
    the baseline profile. INPUT / drift rows may take additional time (or may never
    arrive on a known-broken backend) — we report status but don't block on them.
    Raises only on timeout.
    """
    deadline = time.time() + timeout_s
    while time.time() <= deadline:
        counts = _table_counts()
        if counts is not None:
            n_baseline, n_input, n_drift = counts
            if n_baseline >= 1:
                if n_input >= 2 and n_drift >= 2:
                    print(f"  ✓ refresh complete — baseline cols={n_baseline}, input windows={n_input}, drift windows={n_drift}")
                else:
                    print(f"  baseline cols={n_baseline} input_windows={n_input} drift_windows={n_drift}")
                    print(
                        f"  ⚠ INPUT / drift rows not yet populated. The display cells below will "
                        f"return 0 rows until the refresh writes them — re-run those cells (or the "
                        f"whole notebook) in ~5 min, or watch the dashboard URL above for live updates."
                    )
                return
            print(f"  metric tables exist but no BASELINE rows yet — waiting...")
        else:
            print(f"  metric tables not yet created — waiting...")
        time.sleep(poll_s)
    print(
        f"  ⚠ metric tables did not gain BASELINE rows after {timeout_s}s. The monitor refresh is "
        f"stuck — check the dashboard URL printed above and refresh state via "
        f"`databricks quality-monitors list-refreshes {predictions_table}`. The display cells below "
        f"will print a soft 'not yet readable' warning instead of crashing."
    )


poll_for_metric_tables()


def safe_display(label, sql):
    """Display the result of a query, but degrade gracefully if the metric table
    isn't yet readable. On a healthy refresh this is a no-op; on a slow / stuck
    refresh it prevents the notebook from crashing so you can still see
    the dashboard URL and re-run individual cells later.
    """
    try:
        display(spark.sql(sql))
    except Exception as e:
        print(f"  ⚠ {label}: metric table not yet readable ({type(e).__name__}). Re-run this cell once the refresh completes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Profile Metrics
# MAGIC
# MAGIC The `_profile_metrics` table holds **per-column statistics for each time window** — count, avg, stddev, min, max, percent zeros, percent nulls, quantiles, and classification metrics when a label column is present.

# COMMAND ----------

# Window manifest — confirms we have both recent INPUT windows.
safe_display("window manifest", f"""
    SELECT log_type, window, granularity, model_id
    FROM {profile_metrics_table}
    WHERE column_name = ':table' OR log_type = 'BASELINE'
    GROUP BY log_type, window, granularity, model_id
    ORDER BY log_type, window
""")

# COMMAND ----------

# Numerical + categorical features side by side across windows.
safe_display("feature profile", f"""
    SELECT log_type, window, column_name,
           count, avg, stddev, min, max, percent_zeros
    FROM {profile_metrics_table}
    WHERE column_name IN ('monthly_charges', 'tenure_months', 'contract_type', 'prediction')
      AND slice_key IS NULL
      AND log_type = 'INPUT'
    ORDER BY column_name, window
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Drift Metrics
# MAGIC
# MAGIC The `_drift_metrics` table compares each window against the baseline using six statistical tests:
# MAGIC
# MAGIC | Test | Type | What it measures |
# MAGIC |------|------|-----------------|
# MAGIC | **KS test** | Numerical | Max CDF distance — are distributions shaped differently? |
# MAGIC | **Chi-squared** | Categorical | Frequency differences — has the category mix changed? |
# MAGIC | **PSI** | Both | Population Stability Index — overall distribution shift |
# MAGIC | **Wasserstein** | Numerical | Earth mover's distance |
# MAGIC | **Jensen-Shannon** | Both | Symmetric KL divergence |
# MAGIC | **Total variation** | Both | Max absolute probability difference across bins |

# COMMAND ----------

# Numerical features — these are the ones we drifted.
safe_display("numerical drift", f"""
    SELECT window, column_name,
           ROUND(wasserstein_distance, 4) AS wasserstein,
           ROUND(ks_test.statistic, 4) AS ks_statistic,
           ROUND(ks_test.pvalue, 6) AS ks_pvalue,
           ROUND(population_stability_index, 4) AS psi
    FROM {drift_metrics_table}
    WHERE column_name IN ('monthly_charges', 'total_charges', 'tenure_months')
      AND slice_key IS NULL
      AND window IS NOT NULL
    ORDER BY column_name, window
""")

# COMMAND ----------

# Categorical features (chi² + JS + TV) including the prediction column itself.
safe_display("categorical drift", f"""
    SELECT window, column_name,
           ROUND(chi_squared_test.statistic, 4) AS chi2_statistic,
           ROUND(chi_squared_test.pvalue, 6) AS chi2_pvalue,
           ROUND(js_distance, 4) AS js_distance,
           ROUND(tv_distance, 4) AS tv_distance
    FROM {drift_metrics_table}
    WHERE column_name IN ('contract_type', 'internet_service', 'payment_method', 'prediction')
      AND slice_key IS NULL
      AND window IS NOT NULL
    ORDER BY column_name, window
""")

# COMMAND ----------

# Drift status rollup — ranks features by how much they moved.
safe_display("drift status rollup", f"""
    SELECT column_name, window,
           ROUND(COALESCE(population_stability_index, 0), 4) AS psi,
           ROUND(COALESCE(wasserstein_distance, 0), 4) AS wasserstein,
           ROUND(COALESCE(js_distance, 0), 4) AS js_distance,
           ROUND(COALESCE(ks_test.pvalue, chi_squared_test.pvalue), 6) AS test_pvalue,
           CASE
               WHEN COALESCE(population_stability_index, 0) > 0.2 THEN 'MAJOR DRIFT'
               WHEN COALESCE(population_stability_index, 0) > 0.1 THEN 'MODERATE DRIFT'
               WHEN COALESCE(ks_test.pvalue, chi_squared_test.pvalue, 1) < 0.05 THEN 'SIGNIFICANT'
               ELSE 'STABLE'
           END AS drift_status
    FROM {drift_metrics_table}
    WHERE slice_key IS NULL
      AND window IS NOT NULL
    ORDER BY COALESCE(population_stability_index, 0) DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Model Quality
# MAGIC
# MAGIC Because we configured `label_col="churn_label"`, the monitor automatically computes classification metrics per window from the labels we wrote into the predictions table.

# COMMAND ----------

# precision/recall/f1_score can be either DOUBLE or STRUCT<macro,micro,weighted>
# depending on the monitoring version — detect at runtime and unwrap accordingly.
def _quality_select():
    try:
        schema = spark.table(profile_metrics_table).schema
    except Exception:
        return "*"
    fields = {f.name: f.dataType for f in schema.fields}
    from pyspark.sql.types import StructType
    def col(name):
        dt = fields.get(name)
        if isinstance(dt, StructType):
            sub = {f.name for f in dt.fields}
            picked = "weighted" if "weighted" in sub else next(iter(sub))
            return f"ROUND({name}.{picked}, 4) AS {name}_{picked}"
        return f"ROUND({name}, 4) AS {name}"
    parts = ["window", "ROUND(accuracy_score, 4) AS accuracy"]
    for name in ("precision", "recall", "f1_score"):
        if name in fields:
            parts.append(col(name))
    if "log_loss" in fields:
        parts.append("ROUND(log_loss, 4) AS log_loss")
    return ",\n           ".join(parts)

safe_display("model quality", f"""
    SELECT {_quality_select()}
    FROM {profile_metrics_table}
    WHERE column_name = ':table'
      AND slice_key IS NULL
      AND accuracy_score IS NOT NULL
      AND window IS NOT NULL
    ORDER BY window
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Custom Metric: `high_risk_percentage`
# MAGIC
# MAGIC Jinja SQL template evaluated per window: `100.0 * SUM(CASE WHEN {{prediction_col}} = 1 THEN 1 ELSE 0 END) / COUNT(*)`. After we drift the features and re-score, the model marks more customers high-risk — the metric should climb from the first window to the second.

# COMMAND ----------

safe_display("high_risk_percentage", f"""
    SELECT window,
           ROUND(high_risk_percentage, 2) AS high_risk_pct
    FROM {profile_metrics_table}
    WHERE high_risk_percentage IS NOT NULL
      AND slice_key IS NULL
      AND column_name = ':table'
      AND window IS NOT NULL
    ORDER BY window
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Alerting + Dashboard
# MAGIC
# MAGIC In production the loop is:
# MAGIC ```
# MAGIC Scheduled refresh → drift_metrics table → SQL Alert (threshold check)
# MAGIC    → Notification (email / Slack / PagerDuty) → Workflow (retraining pipeline)
# MAGIC ```
# MAGIC The query below is a **working alert query** — wire it into a Databricks SQL Alert with a "rows returned > 0" condition.

# COMMAND ----------

alert_query = f"""
    SELECT column_name, window,
           ROUND(COALESCE(population_stability_index, 0), 4) AS psi,
           ROUND(COALESCE(ks_test.pvalue, chi_squared_test.pvalue), 6) AS test_pvalue,
           CASE
               WHEN COALESCE(population_stability_index, 0) > 0.2 THEN 'CRITICAL — major distribution shift'
               WHEN COALESCE(population_stability_index, 0) > 0.1 THEN 'WARNING — moderate drift'
               WHEN COALESCE(ks_test.pvalue, chi_squared_test.pvalue, 1) < 0.05 THEN 'ALERT — statistically significant drift'
               ELSE 'OK'
           END AS status
    FROM {drift_metrics_table}
    WHERE slice_key IS NULL
      AND window IS NOT NULL
      AND (
          COALESCE(population_stability_index, 0) > 0.1
          OR COALESCE(ks_test.pvalue, chi_squared_test.pvalue, 1) < 0.05
      )
    ORDER BY COALESCE(population_stability_index, 0) DESC
"""
try:
    alert_results = spark.sql(alert_query)
    display(alert_results)
    print(f"{alert_results.count()} drift alert(s) would fire across the windows above")
except Exception as e:
    print(f"  ⚠ alert query: drift table not yet readable ({type(e).__name__}). Re-run once the refresh completes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | What you built | Details |
# MAGIC |--------------|---------|
# MAGIC | **Baseline table** | Scored training split → `churn_predictions_baseline` (with labels) |
# MAGIC | **Windowed predictions** | One clean overwrite — two recent windows: original features + drifted/re-scored |
# MAGIC | **Monitor** | InferenceLog, monthly granularity, baseline comparison, `label_col` set up front so model-quality metrics populate on the first refresh |
# MAGIC | **Drift detection** | KS, chi-squared, PSI, Wasserstein, JS divergence, total variation |
# MAGIC | **Model quality** | Accuracy, precision, recall, F1 — computed automatically because `label_col` was configured |
# MAGIC | **Custom metric** | `high_risk_percentage` — business KPI per window |
# MAGIC | **Alerting** | SQL query with PSI / p-value thresholds — drop into a Databricks SQL Alert |
# MAGIC | **Dashboard** | Auto-generated Lakeview dashboard (URL printed above) |
# MAGIC
# MAGIC ### Databricks Lakehouse Monitoring vs. Third-Party Tools
# MAGIC
# MAGIC | Capability | Lakehouse Monitoring | Evidently / Great Expectations |
# MAGIC |-----------|---------------------|-------------------------------|
# MAGIC | **Setup** | One API call — no infrastructure | Separate service / container to manage |
# MAGIC | **Data stays in place** | Reads from Delta tables directly | Requires data export or connector |
# MAGIC | **Drift tests** | 6 built-in (KS, chi², PSI, Wasserstein, JS, TV) | Varies by tool |
# MAGIC | **Custom metrics** | Jinja SQL templates — any SQL expression | Python functions |
# MAGIC | **Dashboard** | Auto-generated Lakeview | Manual setup |
# MAGIC | **Alerting** | Native SQL Alerts → email/Slack/PagerDuty | Separate alerting pipeline |
# MAGIC | **GenAI monitoring** | Text quality, toxicity, topic drift | Varies |
# MAGIC | **Cost** | Serverless SQL compute for refresh | Infrastructure + license costs |
# MAGIC | **Governance** | Unity Catalog lineage built-in | External metadata management |
