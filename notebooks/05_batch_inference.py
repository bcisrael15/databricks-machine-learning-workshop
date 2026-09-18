# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Customer Churn — End-to-End MLflow
# MAGIC ## Notebook 05 — Batch Inference
# MAGIC
# MAGIC
# MAGIC You'll score the test split of customers and write a `churn_predictions` table —
# MAGIC the pre-computed-scores pattern, and the table notebook 06 monitors for drift.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.6.0 mlflow==3.8.1 scikit-learn==1.6.1 "numpy<2" -q
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

import mlflow
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Assemble Inference Data
# MAGIC
# MAGIC Join the test-split customers to the offline feature table to get the feature vector each row needs.

# COMMAND ----------

test_labels = spark.table("churn_labels").filter("split = 'test'").select("customer_id")
features = spark.table(feature_table_name).drop("update_timestamp")

inference_df = (
    test_labels
    .join(features, on="customer_id", how="inner")
    # Compute the on-demand feature `avg_price_increase` via the UDF created in nb 02.
    .withColumn(
        "avg_price_increase",
        F.expr(f"{catalog}.{schema}.avg_price_increase(monthly_charges, tenure_months)")
    )
)
print(f"Test customers to score: {inference_df.count()}")
display(inference_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the Champion Model and Score
# MAGIC
# MAGIC Loading by **alias** (`models:/<name>@Champion`) rather than a fixed version number means
# MAGIC this job always scores whatever model is currently promoted — you can roll a new version
# MAGIC to Champion without changing any downstream code.

# COMMAND ----------

model_uri = f"models:/{model_name}@Champion"
model = mlflow.sklearn.load_model(model_uri)
feature_cols = [c for c in inference_df.columns if c != "customer_id"]

# Bring features to pandas for scoring (test set is small — fine for the workshop)
pdf = inference_df.toPandas()
X = pdf[feature_cols]

# Cast prediction to float — monitoring requires prediction and label types to match (both DOUBLE).
pdf["prediction"] = model.predict(X).astype("float64")
pdf["churn_probability"] = model.predict_proba(X)[:, 1]

# Keep features alongside predictions — Notebook 06 (monitoring) reads them
# to compute drift metrics on feature distributions.
output_cols = ["customer_id"] + feature_cols + ["prediction", "churn_probability"]
predictions_df = spark.createDataFrame(pdf[output_cols])
display(predictions_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write the Predictions Table

# COMMAND ----------

(
    predictions_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(predictions_table_name)
)

print(f"✓ Predictions saved to {predictions_table_name}")
print(f"  Total predictions: {spark.table(predictions_table_name).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Joined the test split to the **offline feature table** to build the inference DataFrame
# MAGIC - Loaded the **Champion** model from Unity Catalog and scored it
# MAGIC - Wrote `churn_predictions` to UC — notebook 06 monitors this table for drift
# MAGIC
# MAGIC **Next**: [06 Monitoring →](./06_monitoring)
