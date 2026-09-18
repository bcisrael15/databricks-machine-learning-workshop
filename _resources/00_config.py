# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Configuration
# MAGIC
# MAGIC Loaded by every notebook via `%run ../_resources/00_config`. It **reads** the
# MAGIC `catalog` / `schema` widgets that each notebook defines at its top, and derives the
# MAGIC table / model / endpoint names from them. (The widgets live in each notebook — not
# MAGIC here — because widgets created inside a `%run` don't render on serverless.)

# COMMAND ----------

import re

# Read the widgets the calling notebook defined at its top.
catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()

# Per-user serving endpoint (workspace-global name → must be unique per participant).
_current_user = spark.sql("SELECT current_user()").collect()[0][0]
_user_slug = re.sub(r"[^a-z0-9_]", "_", _current_user.split("@")[0].lower()).strip("_") or "user"
churn_model_serving_endpoint = f"churn-model-{_user_slug}".replace("_", "-")

# Derived references
feature_table_name = f"{catalog}.{schema}.churn_feature_table"
model_name = f"{catalog}.{schema}.churn_model"
predictions_table_name = f"{catalog}.{schema}.churn_predictions"

# COMMAND ----------

# Set default catalog and schema for SQL (skip if not yet created by setup)
try:
    spark.sql(f"USE CATALOG `{catalog}`")
    spark.sql(f"USE SCHEMA `{schema}`")
except Exception:
    print(f"Note: catalog '{catalog}' or schema '{schema}' not yet created — run 01_setup_slim first")

print(f"User:             {_current_user}")
print(f"Catalog:          {catalog}   (shared)")
print(f"Schema:           {schema}   (yours)")
print(f"Feature table:    {feature_table_name}")
print(f"Model:            {model_name}")
print(f"Serving endpoint: {churn_model_serving_endpoint}")
