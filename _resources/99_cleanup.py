# Databricks notebook source
# MAGIC %md
# MAGIC # Cleanup (your own assets only)
# MAGIC
# MAGIC Tears down **everything you created** in the speedrun: your schema (CASCADE),
# MAGIC your serving endpoint, the monitor on your predictions table, and your MLflow
# MAGIC experiments (including the AutoML run from 03b). It only touches assets keyed to
# MAGIC your username — other
# MAGIC participants' schemas are untouched, and the shared catalog is **not** dropped.
# MAGIC
# MAGIC Requires typing `DELETE` into the confirmation widget.

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

# MAGIC %run ./00_config

# COMMAND ----------

dbutils.widgets.text("confirm", "", "Type DELETE to confirm")
confirm = dbutils.widgets.get("confirm")
if confirm != "DELETE":
    raise Exception("Type DELETE into the 'confirm' widget above to run cleanup.")
print("Confirmed — cleaning up your assets.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.tracking import MlflowClient

w = WorkspaceClient()

# COMMAND ----------

# 1. Delete the serving endpoint (if it exists)
try:
    w.serving_endpoints.delete(name=churn_model_serving_endpoint)
    print(f"✓ Deleted serving endpoint {churn_model_serving_endpoint}")
except Exception as e:
    print(f"  serving endpoint {churn_model_serving_endpoint}: {e}")

# COMMAND ----------

# 2. Delete the Lakehouse Monitor AND its auto-generated Lakeview dashboard.
#    (Deleting the monitor does NOT remove the dashboard — grab its id first, then trash it.)
dashboard_id = None
try:
    dashboard_id = getattr(w.quality_monitors.get(table_name=predictions_table_name), "dashboard_id", None)
except Exception:
    pass

try:
    w.quality_monitors.delete(table_name=predictions_table_name)
    print(f"✓ Deleted monitor on {predictions_table_name}")
except Exception as e:
    print(f"  monitor on {predictions_table_name}: {e}")

if dashboard_id:
    try:
        w.api_client.do("DELETE", f"/api/2.0/lakeview/dashboards/{dashboard_id}")
        print(f"✓ Trashed monitor dashboard {dashboard_id}")
    except Exception as e:
        print(f"  dashboard {dashboard_id}: {e}")

# COMMAND ----------

# 3. Drop your schema (CASCADE removes tables, feature table, models, functions —
#    including the churn_model_automl model + automl_training_input table from 03b)
spark.sql(f"DROP SCHEMA IF EXISTS {catalog}.{schema} CASCADE")
print(f"✓ Dropped schema {catalog}.{schema}")

# COMMAND ----------

# 4. Delete your MLflow experiments (03 → ml-ai-workshop-churn, 03b AutoML → churn-automl)
for experiment_path in (
    f"/Users/{_current_user}/ml-ai-workshop-churn",
    f"/Users/{_current_user}/churn-automl",
):
    try:
        exp = mlflow.get_experiment_by_name(experiment_path)
        if exp:
            MlflowClient().delete_experiment(exp.experiment_id)
            print(f"✓ Deleted experiment {experiment_path}")
        else:
            print(f"  no experiment at {experiment_path}")
    except Exception as e:
        print(f"  experiment {experiment_path}: {e}")

# COMMAND ----------

print("Cleanup complete — your speedrun assets are gone. Re-run 01_setup_slim to start over.")
