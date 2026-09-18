# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 03b — AutoML Hyperparameter Tuning
# MAGIC
# MAGIC An alternative to the manual LightGBM + Optuna flow in notebook 03. Databricks **AutoML**
# MAGIC automatically explores multiple algorithms and hyperparameter spaces, producing a
# MAGIC leaderboard of tuned models — with zero boilerplate code. The best model is registered
# MAGIC under `churn_model_automl` in the same schema.
# MAGIC
# MAGIC > **⚠️ Compute:** AutoML runs on a **classic ML Runtime cluster only — not serverless.**
# MAGIC > Attach an ML Runtime cluster before running this notebook. (The rest of the workshop
# MAGIC > uses serverless; `03b` is the one exception.)

# COMMAND ----------

# DBTITLE 1,Install AutoML runtime
# AutoML requires a CLASSIC ML Runtime cluster (it does NOT run on serverless).
# On DBR < 18.0 ML, databricks.automl is pre-installed. On DBR 18.0 ML+ it's no longer
# bundled, so install it from PyPI (harmless on older runtimes):
%pip install databricks-automl-runtime mlflow -q
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widget setup
import re
_u = spark.sql("SELECT current_user()").collect()[0][0]
_slug = re.sub(r"[^a-z0-9_]", "_", _u.split("@")[0].lower()).strip("_") or "user"
dbutils.widgets.text("catalog", "main", "1. Catalog (shared)")
dbutils.widgets.text("schema", f"churn_workshop_{_slug}", "2. Schema (yours)")

# COMMAND ----------

# DBTITLE 1,Load config
# MAGIC %run ../_resources/00_config

# COMMAND ----------

# DBTITLE 1,Imports
import databricks.automl
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

# --- AWS GovCloud / non-standard AWS partitions only ---
# On GovCloud, MLflow can't auto-detect the Unity Catalog storage bucket's region, so AutoML's
# artifact/model writes can fail with:
#   "Unable to determine the region for S3 bucket ... HeadBucket ... 400 Bad Request"
# If you hit that, set your region here before running the AutoML cells, e.g.:
#   import os; os.environ["AWS_DEFAULT_REGION"] = "us-gov-west-1"
# Leave this commented on commercial AWS / Azure — it isn't needed there.

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

# New model name for the AutoML route
automl_model_name = f"{catalog}.{schema}.churn_model_automl"
print(f"AutoML model will be registered as: {automl_model_name}")

# COMMAND ----------

# DBTITLE 1,Prepare training data
# MAGIC %md
# MAGIC ## Prepare Training Data
# MAGIC
# MAGIC Same data pipeline as notebook 03 — join the feature table to `churn_labels` for the
# MAGIC train + val splits. AutoML expects a single Spark DataFrame with the target column included.

# COMMAND ----------

# DBTITLE 1,Build training DataFrame
# Load labels (train + val splits, same as notebook 03)
labels_df = spark.table("churn_labels").filter("split IN ('train', 'val')")

# Load feature table and compute the on-demand feature
features = (
    spark.table(feature_table_name)
    .drop("update_timestamp")
    .withColumn(
        "avg_price_increase",
        F.expr(f"{catalog}.{schema}.avg_price_increase(monthly_charges, tenure_months)")
    )
)

# Join features with labels — AutoML needs the target in the same DataFrame
training_df = (
    labels_df.select("customer_id", "churn")
    .join(features, on="customer_id", how="inner")
    .drop("customer_id")  # AutoML doesn't need the ID column
)

# Persist to UC so the AutoML job (separate compute) can access it reliably
automl_input_table = f"{catalog}.{schema}.automl_training_input"
training_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(automl_input_table)

print(f"Training rows: {spark.table(automl_input_table).count()}, columns: {len(training_df.columns)}")
print(f"Saved to: {automl_input_table}")
display(training_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Run AutoML header
# MAGIC %md
# MAGIC ## Run AutoML Classification
# MAGIC
# MAGIC One call kicks off automated exploration across multiple algorithms (LightGBM, XGBoost,
# MAGIC Logistic Regression, Decision Trees, Random Forest) and hyperparameter configurations.
# MAGIC All trials are logged to a dedicated MLflow experiment.

# COMMAND ----------

# DBTITLE 1,Run AutoML classify
# Set experiment for AutoML
experiment_path = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/churn-automl"
mlflow.set_experiment(experiment_path)

summary = databricks.automl.classify(
    dataset=automl_input_table,
    target_col="churn",
    primary_metric="f1",
    timeout_minutes=10,
    max_trials=20,
)

print(f"\nBest trial metric (F1): {summary.best_trial.metrics['test_f1_score']:.4f}")
print(f"Best trial notebook: {summary.best_trial.notebook_path}")
print(f"Best run ID: {summary.best_trial.mlflow_run_id}")

# COMMAND ----------

# DBTITLE 1,Register model header
# MAGIC %md
# MAGIC ## Register Best Model to Unity Catalog
# MAGIC
# MAGIC Register the winning AutoML trial under a new model name (`churn_model_automl`) in the
# MAGIC same schema, so it lives alongside the manual LightGBM model from notebook 03.

# COMMAND ----------

# DBTITLE 1,Register AutoML model
best_run_id = summary.best_trial.mlflow_run_id

# Find the model artifact path from the best run
best_run = mlflow_client.get_run(best_run_id)
artifact_uri = f"runs:/{best_run_id}/model"

# Register to Unity Catalog under the new name
mv = mlflow.register_model(model_uri=artifact_uri, name=automl_model_name)
print(f"\u2713 Registered: {automl_model_name} v{mv.version}")

# Tag with metrics for easy comparison
best_metrics = summary.best_trial.metrics
for k, v in best_metrics.items():
    mlflow_client.set_model_version_tag(
        name=automl_model_name, version=mv.version, key=k, value=f"{v:.4f}"
    )

mlflow_client.update_model_version(
    name=automl_model_name,
    version=mv.version,
    description=(
        f"AutoML best model — F1={best_metrics.get('test_f1_score', 0):.4f}. "
        f"Trained via databricks.automl.classify() with up to 20 trials."
    ),
)

# Set aliases
mlflow_client.set_registered_model_alias(name=automl_model_name, alias="Challenger", version=mv.version)
mlflow_client.set_registered_model_alias(name=automl_model_name, alias="Champion", version=mv.version)
print(f"\u2713 Version {mv.version} aliased as Champion + Challenger")

# COMMAND ----------

# DBTITLE 1,Compare models header
# MAGIC %md
# MAGIC ## Quick Comparison: Manual vs AutoML
# MAGIC
# MAGIC Pull the Champion version from both the manual model and the AutoML model to compare headline metrics.

# COMMAND ----------

# DBTITLE 1,Compare models
import pandas as pd

rows = []
for name, label in [(model_name, "Manual (LightGBM+Optuna)"), (automl_model_name, "AutoML")]:
    try:
        mv_info = mlflow_client.get_model_version_by_alias(name, "Champion")
        run = mlflow_client.get_run(mv_info.run_id)
        metrics = run.data.metrics
        rows.append({
            "Model": label,
            "F1": round(metrics.get("f1_score", metrics.get("test_f1_score", 0)), 4),
            "AUC": round(metrics.get("roc_auc", metrics.get("test_roc_auc", 0)), 4),
            "Version": mv_info.version,
        })
    except Exception as e:
        rows.append({"Model": label, "F1": "N/A", "AUC": "N/A", "Version": str(e)})

display(pd.DataFrame(rows))