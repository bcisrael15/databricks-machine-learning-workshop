# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Customer Churn — End-to-End MLflow
# MAGIC ## Notebook 02 — Feature Engineering with Feature Store
# MAGIC
# MAGIC
# MAGIC You'll create a **feature table** in Unity Catalog that joins customer profiles with aggregated service ticket features.
# MAGIC
# MAGIC Key concepts:
# MAGIC - Feature Engineering Client (`databricks.feature_engineering`)
# MAGIC - Feature tables with primary keys and timeseries columns
# MAGIC - On-demand feature functions

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering==0.14.0 databricks-sdk>=0.50.0
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

from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Features from Customers + Service Tickets
# MAGIC
# MAGIC You'll aggregate service ticket data per customer and join with customer profiles to create the feature table.

# COMMAND ----------

# Load source tables
customers = spark.table("customers")
tickets = spark.table("service_tickets")

# COMMAND ----------

# Aggregate ticket features per customer
ticket_features = (
    tickets
    .groupBy("customer_id")
    .agg(
        F.count("ticket_id").alias("total_tickets"),
        F.sum(F.when(F.col("category") == "billing", 1).otherwise(0)).alias("billing_tickets"),
        F.sum(F.when(F.col("category") == "technical", 1).otherwise(0)).alias("technical_tickets"),
        F.sum(F.when(F.col("category") == "cancellation", 1).otherwise(0)).alias("cancellation_tickets"),
        F.sum(F.when(F.col("priority").isin("high", "critical"), 1).otherwise(0)).alias("high_priority_tickets"),
        F.avg("resolution_time_hours").alias("avg_resolution_hours"),
        F.max("created_date").alias("last_ticket_date"),
    )
)

display(ticket_features.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Count Optional Services
# MAGIC
# MAGIC You'll use a **Pandas UDF** to compute how many optional services each customer has enabled.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf

optional_service_cols = [
    "streaming_tv", "streaming_movies", "online_security",
    "online_backup", "device_protection", "tech_support"
]

@pandas_udf(IntegerType())
def count_optional_services(
    streaming_tv: pd.Series, streaming_movies: pd.Series,
    online_security: pd.Series, online_backup: pd.Series,
    device_protection: pd.Series, tech_support: pd.Series,
) -> pd.Series:
    cols = [streaming_tv, streaming_movies, online_security, online_backup, device_protection, tech_support]
    return sum((col == "Yes").astype(int) for col in cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the Feature DataFrame

# COMMAND ----------

# Join customers with ticket aggregates and compute derived features
feature_df = (
    customers
    .join(ticket_features, on="customer_id", how="left")
    .fillna(0, subset=["total_tickets", "billing_tickets", "technical_tickets",
                        "cancellation_tickets", "high_priority_tickets", "avg_resolution_hours"])
    .withColumn(
        "num_optional_services",
        count_optional_services(*[F.col(c) for c in optional_service_cols])
    )
    .withColumn("update_timestamp", F.current_timestamp())
    .select(
        "customer_id",
        "tenure_months",
        "contract_type",
        "monthly_charges",
        "total_charges",
        "payment_method",
        "internet_service",
        "phone_service",
        "paperless_billing",
        "partner",
        "dependents",
        "senior_citizen",
        "num_optional_services",
        "total_tickets",
        "billing_tickets",
        "technical_tickets",
        "cancellation_tickets",
        "high_priority_tickets",
        "avg_resolution_hours",
        "update_timestamp",
    )
)

print(f"Feature table rows: {feature_df.count()}")
display(feature_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Table in Unity Catalog
# MAGIC
# MAGIC The `FeatureEngineeringClient` registers this as a managed UC feature table (a Delta
# MAGIC table with a primary key) with lineage tracking.
# MAGIC
# MAGIC **Why a feature table vs. a plain Delta table:** features become reusable across models,
# MAGIC the feature-lookup logic can be packaged into the trained model so training and serving
# MAGIC stay consistent (no train/serve skew), and Unity Catalog records which features fed which
# MAGIC model.

# COMMAND ----------

# Drop if exists (for re-runnability)
try:
    fe.drop_table(name=feature_table_name)
    print(f"Dropped existing table {feature_table_name}")
except Exception:
    pass

# COMMAND ----------

fe.create_table(
    name=feature_table_name,
    primary_keys=["customer_id"],
    df=feature_df,
    description="Customer churn prediction features: demographics, service profile, and ticket aggregates",
)

print(f"✓ Feature table created: {feature_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## On-Demand Feature Function
# MAGIC

# COMMAND ----------

# Drop existing function (may be a different language, preventing CREATE OR REPLACE)
spark.sql(f"DROP FUNCTION IF EXISTS {catalog}.{schema}.avg_price_increase")

spark.sql(f"""
CREATE FUNCTION {catalog}.{schema}.avg_price_increase(monthly_charges DOUBLE, tenure_months BIGINT)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'Computes the average monthly price increase over tenure. Used as an on-demand feature at training and serving time.'
AS $$
return monthly_charges / tenure_months if tenure_months and tenure_months > 0 else 0.0
$$
""")
print(f"✓ On-demand feature function created: {catalog}.{schema}.avg_price_increase")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC You created:
# MAGIC - A **feature table** (`churn_feature_table`) with customer demographics + ticket aggregates
# MAGIC - An **on-demand feature function** (`avg_price_increase`) for dynamic computation
# MAGIC
# MAGIC Both are registered in **Unity Catalog** with full lineage tracking.
# MAGIC
# MAGIC **Next**: [03 Train Model →](./03_train_and_register_model)
