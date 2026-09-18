# Bundled data — for the `volume_csv` setup route

Three CSVs — the exact tables the default `generate` route produces (seed 42):

| File | Rows | Loads into |
|------|------|-----------|
| `customers.csv` | 5,000 | `<catalog>.<schema>.customers` |
| `service_tickets.csv` | 15,000 | `<catalog>.<schema>.service_tickets` |
| `churn_labels.csv` | 5,000 | `<catalog>.<schema>.churn_labels` |

**You only need these if you run `01_setup_slim` with `data_source = volume_csv`** (e.g. on a
locked-down / air-gapped workspace). Otherwise the default `generate` route builds the same
data in-notebook with no pip and these files are unused.

## Using them

1. Run `01_setup_slim` once — it creates your schema and a `raw` Volume at
   `/Volumes/<catalog>/<schema>/raw`.
2. Upload these three CSVs to that Volume:
   - **UI:** Catalog Explorer ▸ the `raw` Volume ▸ **Upload to this volume**.
   - **CLI:** `databricks fs cp data/<file>.csv dbfs:/Volumes/<catalog>/<schema>/raw/<file>.csv`
3. Re-run the `volume_csv` load cell in `01_setup_slim`. It reads the CSVs with
   `header=true, inferSchema=true` and writes the managed tables.

*Generated data — synthetic telecom customers, tickets, and churn labels. No real PII.*
