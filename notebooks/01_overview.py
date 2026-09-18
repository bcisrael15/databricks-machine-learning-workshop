# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Customer Churn — End-to-End MLflow
# MAGIC ## Notebook 01 — Overview & Data Exploration
# MAGIC
# MAGIC
# MAGIC An end-to-end example of the **full MLflow / MLOps lifecycle** on a synthetic telecom
# MAGIC customer-churn dataset. You'll touch every stage, from features to monitoring:
# MAGIC
# MAGIC | Notebook | Topic |
# MAGIC |----------|-------|
# MAGIC | **01 Overview** | ML Runtime, data exploration |
# MAGIC | 02 Feature Engineering | Feature Store with Unity Catalog |
# MAGIC | 03 Train Model | LightGBM + Optuna, MLflow tracking, `mlflow.evaluate`, registry + aliases |
# MAGIC | 04 Model Serving | Real-time model serving endpoint |
# MAGIC | 05 Batch Inference | Batch scoring with the Champion model |
# MAGIC | 06 Monitoring | Lakehouse Monitoring for drift detection |
# MAGIC
# MAGIC ### The pipeline you'll build
# MAGIC ```
# MAGIC customers ──┐
# MAGIC             ├──→ feature_table ──→ model ──→ serving_endpoint (real-time)
# MAGIC service_tickets ┘                    │
# MAGIC                                      └──→ batch predictions ──→ monitoring
# MAGIC ```

# COMMAND ----------

import re
_u = spark.sql("SELECT current_user()").collect()[0][0]
_slug = re.sub(r"[^a-z0-9_]", "_", _u.split("@")[0].lower()).strip("_") or "user"
dbutils.widgets.text("catalog", "main", "1. Catalog (shared)")
dbutils.widgets.text("schema", f"churn_workshop_{_slug}", "2. Schema (yours)")

# COMMAND ----------

# MAGIC %run ../_resources/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Runtime
# MAGIC
# MAGIC This cluster runs the **Databricks ML Runtime**, which comes pre-installed with popular ML libraries.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore the Data
# MAGIC
# MAGIC Our workshop uses a **synthetic telecom dataset** with realistic churn correlations.
# MAGIC All tables live in Unity Catalog under `{catalog}.{schema}`.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Quick overview of available tables
# MAGIC SHOW TABLES

# COMMAND ----------

# Load customers table
customers_df = spark.table("customers")
display(customers_df.limit(10))

# COMMAND ----------

print(f"Total customers: {customers_df.count()}")
print(f"\nSchema:")
customers_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Churn Distribution
# MAGIC Our target variable is `churn` (Yes/No). The dataset has a ~33% churn rate — elevated on purpose so both classes are well represented for training.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT churn, COUNT(*) as count,
# MAGIC        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
# MAGIC FROM customers
# MAGIC GROUP BY churn

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Churn Signals

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Contract type vs churn
# MAGIC SELECT contract_type, churn, COUNT(*) as count
# MAGIC FROM customers
# MAGIC GROUP BY contract_type, churn
# MAGIC ORDER BY contract_type, churn

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Tenure distribution by churn status
# MAGIC SELECT churn,
# MAGIC        ROUND(AVG(tenure_months), 1) as avg_tenure,
# MAGIC        ROUND(AVG(monthly_charges), 2) as avg_monthly_charges,
# MAGIC        ROUND(AVG(total_charges), 2) as avg_total_charges
# MAGIC FROM customers
# MAGIC GROUP BY churn

# COMMAND ----------

# MAGIC %md
# MAGIC ### Service Tickets

# COMMAND ----------

tickets_df = spark.table("service_tickets")
print(f"Total service tickets: {tickets_df.count()}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Ticket category distribution
# MAGIC SELECT category, COUNT(*) as count
# MAGIC FROM service_tickets
# MAGIC GROUP BY category
# MAGIC ORDER BY count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Ticket volume by churn status (do churners file more tickets?)
# MAGIC SELECT c.churn,
# MAGIC        ROUND(COUNT(t.ticket_id) * 1.0 / COUNT(DISTINCT c.customer_id), 1) as avg_tickets_per_customer
# MAGIC FROM customers c
# MAGIC LEFT JOIN service_tickets t ON c.customer_id = t.customer_id
# MAGIC GROUP BY c.churn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We have a rich dataset with clear churn signals:
# MAGIC - **Month-to-month** contracts churn at much higher rates
# MAGIC - **Short tenure** and **high charges** correlate with churn
# MAGIC - **Churners file more tickets**, especially in billing and cancellation categories
# MAGIC
# MAGIC **Next**: [02 Feature Engineering →](./02_feature_engineering)
