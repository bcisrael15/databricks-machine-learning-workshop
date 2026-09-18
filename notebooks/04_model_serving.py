# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Customer Churn — End-to-End MLflow
# MAGIC ## Notebook 04 — Real-Time Model Serving
# MAGIC
# MAGIC
# MAGIC You'll deploy the Champion model to a **Model Serving** endpoint and call it from a REST client.
# MAGIC
# MAGIC Key concepts:
# MAGIC - Model Serving endpoints with scale-to-zero
# MAGIC - REST invocation with a feature-vector payload
# MAGIC - Pattern: the caller assembles the feature vector (from app state, a UC table, or a vector lookup)

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.102.0 mlflow==3.8.1 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re
_u = spark.sql("SELECT current_user()").collect()[0][0]
_slug = re.sub(r"[^a-z0-9_]", "_", _u.split("@")[0].lower()).strip("_") or "user"
dbutils.widgets.text("catalog", "main", "1. Catalog (shared)")
dbutils.widgets.text("schema", f"churn_workshop_{_slug}", "2. Schema (yours)")

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
import mlflow
from mlflow.tracking import MlflowClient
from datetime import timedelta

w = WorkspaceClient()
mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Look Up the Champion Model

# COMMAND ----------

champion_info = mlflow_client.get_model_version_by_alias(model_name, "Champion")
champion_version = champion_info.version
print(f"Deploying Champion model version: {champion_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the Serving Endpoint
# MAGIC
# MAGIC The endpoint serves the registered model directly — the caller passes in a feature vector at inference time.
# MAGIC With `scale_to_zero_enabled=True` the endpoint spins down to zero compute when idle.

# COMMAND ----------

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        entity_version=str(champion_version),
        scale_to_zero_enabled=True,
        workload_size="Small",
    )
]

try:
    w.serving_endpoints.create_and_wait(
        name=churn_model_serving_endpoint,
        config=EndpointCoreConfigInput(
            name=churn_model_serving_endpoint,
            served_entities=served_entities,
        ),
        timeout=timedelta(minutes=45),
    )
    print(f"✓ Serving endpoint '{churn_model_serving_endpoint}' created and ready")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Endpoint already exists, pushing config update...")
        w.serving_endpoints.update_config_and_wait(
            name=churn_model_serving_endpoint,
            served_entities=served_entities,
            timeout=timedelta(minutes=45),
        )
        print(f"✓ Serving endpoint '{churn_model_serving_endpoint}' updated and ready")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build a Test Payload
# MAGIC
# MAGIC We pull a few rows from the feature table to use as model inputs. In production the caller would
# MAGIC assemble these features however it likes — from a UC table, a cache, or app-side state.

# COMMAND ----------

from pyspark.sql import functions as F

sample_sdf = (
    spark.table(feature_table_name)
    .limit(5)
    .withColumn(
        "avg_price_increase",
        F.expr(f"{catalog}.{schema}.avg_price_increase(monthly_charges, tenure_months)")
    )
)
feature_cols = [c for c in sample_sdf.columns if c not in ("customer_id", "update_timestamp")]

sample_pdf = sample_sdf.toPandas()
sample_customer_ids = sample_pdf["customer_id"].tolist()
feature_records = sample_pdf[feature_cols].to_dict(orient="records")

print(f"Sample customers: {sample_customer_ids}")
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Invoke the Endpoint via REST

# COMMAND ----------

import requests
import json

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

payload = {"dataframe_records": feature_records}

response = requests.post(
    f"https://{workspace_url}/serving-endpoints/{churn_model_serving_endpoint}/invocations",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=payload,
)

print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
response.raise_for_status()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Pretty-Print Predictions

# COMMAND ----------

if response.status_code == 200:
    predictions = response.json()["predictions"]
    for cid, pred in zip(sample_customer_ids, predictions):
        print(f"  {cid}: prediction={pred}")
else:
    print(f"Error: {response.status_code} — {response.text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We deployed the churn model for real-time inference:
# MAGIC - **Model Serving Endpoint** serves the Champion model via REST API with scale-to-zero
# MAGIC - The endpoint expects a **feature vector** per request — the caller supplies features
# MAGIC - Pattern: pre-compute features in a UC table (or assemble them client-side) and pass to the endpoint
# MAGIC
# MAGIC A common alternative to real-time serving is **pre-computing scores in batch** and reading
# MAGIC them from a table — which is exactly what notebook 05 does next.
# MAGIC
# MAGIC **Next**: [05 Batch Inference →](./05_batch_inference)
