# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Workshop Setup (slim, per-participant)
# MAGIC
# MAGIC **Run this once, top to bottom.** It creates *your* schema in the shared catalog
# MAGIC (set via the widgets) and loads the three tables the MLflow speedrun needs.
# MAGIC
# MAGIC | Table | Rows | Contents |
# MAGIC |-------|------|----------|
# MAGIC | `customers` | ~5,000 | Customer profiles + churn label |
# MAGIC | `service_tickets` | ~15,000 | Support tickets per customer |
# MAGIC | `churn_labels` | ~5,000 | Target + train/val/test split |
# MAGIC
# MAGIC ### Two data routes (pick one with the `data_source` widget)
# MAGIC - **`generate`** *(default)* — builds the data in-notebook. **No `pip`, no internet**
# MAGIC   required — pure pandas/numpy that ship with the runtime.
# MAGIC - **`volume_csv`** — loads the bundled CSVs from a **Volume**. Use this on
# MAGIC   locked-down workspaces, or any time you'd rather ship data than generate it.
# MAGIC   You upload the three CSVs (in the repo's `data/` folder) to the Volume first —
# MAGIC   instructions appear below.

# COMMAND ----------

import re
_u = spark.sql("SELECT current_user()").collect()[0][0]
_slug = re.sub(r"[^a-z0-9_]", "_", _u.split("@")[0].lower()).strip("_") or "user"
dbutils.widgets.text("catalog", "main", "1. Catalog (shared)")
dbutils.widgets.text("schema", f"churn_workshop_{_slug}", "2. Schema (yours)")

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

dbutils.widgets.dropdown("data_source", "generate", ["generate", "volume_csv"], "3. Data source")
data_source = dbutils.widgets.get("data_source")
print(f"Data source: {data_source}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create your schema (+ a Volume for the CSV route)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
spark.sql(f"USE CATALOG `{catalog}`")
spark.sql(f"USE SCHEMA `{schema}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.raw")
raw_volume = f"/Volumes/{catalog}/{schema}/raw"
print(f"✓ Schema ready: {catalog}.{schema}")
print(f"✓ Volume ready: {raw_volume}")

# COMMAND ----------

# MAGIC %run ./data_generators

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2A. Route `generate` — build the data in-notebook (no pip)
# MAGIC Runs only when `data_source = generate`.

# COMMAND ----------

TABLES = ["customers", "service_tickets", "churn_labels"]

if data_source == "generate":
    customers_pdf = generate_customers(n=5000, seed=42)
    print(f"Generated {len(customers_pdf)} customers, churn rate: "
          f"{customers_pdf['churn'].value_counts(normalize=True)['Yes']:.1%}")
    spark.createDataFrame(customers_pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.customers")

    tickets_pdf = generate_service_tickets(customers_pdf, n=15000, seed=42)
    spark.createDataFrame(tickets_pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.service_tickets")

    labels_pdf = generate_churn_labels(customers_pdf, seed=42)
    spark.createDataFrame(labels_pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.churn_labels")
    print("✓ Generated and wrote all three tables")
else:
    print("Skipping generate route (data_source != 'generate').")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B. Route `volume_csv` — load bundled CSVs from the Volume
# MAGIC Runs only when `data_source = volume_csv`. **Upload the three CSVs first:**
# MAGIC
# MAGIC - **UI:** Catalog Explorer ▸ your `raw` Volume ▸ **Upload to this volume** ▸ drop
# MAGIC   `customers.csv`, `service_tickets.csv`, `churn_labels.csv` (from the repo's
# MAGIC   `data/` folder).
# MAGIC - **CLI:** `databricks fs cp data/<file>.csv dbfs:/Volumes/<catalog>/<schema>/raw/<file>.csv`
# MAGIC
# MAGIC Then run the cell below. It's re-runnable — if a file is missing it tells you what to
# MAGIC upload and stops without erroring.

# COMMAND ----------

if data_source == "volume_csv":
    try:
        present = {f.name for f in dbutils.fs.ls(raw_volume)}
    except Exception:
        present = set()
    missing = [f"{t}.csv" for t in TABLES if f"{t}.csv" not in present]
    if missing:
        raise Exception(
            f"Upload these to {raw_volume} first, then re-run this cell: {missing}. "
            f"(Catalog Explorer ▸ the 'raw' volume ▸ Upload to this volume, or `databricks fs cp`.)"
        )
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType
    for t in TABLES:
        df = (spark.read
              .option("header", True).option("inferSchema", True)
              .csv(f"{raw_volume}/{t}.csv"))
        # Match the generate route's schema exactly: pandas int64 -> Spark BIGINT, but CSV
        # inferSchema yields 32-bit INT. Upcast so the on-demand feature function (which
        # declares BIGINT params, e.g. tenure_months) and all downstream types line up.
        for fld in df.schema.fields:
            if isinstance(fld.dataType, IntegerType):
                df = df.withColumn(fld.name, F.col(fld.name).cast("bigint"))
        (df.write.mode("overwrite").option("overwriteSchema", "true")
            .saveAsTable(f"{catalog}.{schema}.{t}"))
        print(f"✓ Loaded {t} from CSV")
else:
    print("Skipping volume_csv route (data_source != 'volume_csv').")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verify

# COMMAND ----------

for t in TABLES:
    print(f"  {t}: {spark.table(f'{catalog}.{schema}.{t}').count()} rows")
display(spark.sql("SELECT churn, COUNT(*) AS n FROM customers GROUP BY churn"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup complete
# MAGIC
# MAGIC Your schema is provisioned and owned by you. **Next**: work through the notebooks in
# MAGIC `notebooks/` in order, `01` → `06`. Each has the same `catalog` / `schema` widgets —
# MAGIC just confirm they point at your sandbox before Running All.
