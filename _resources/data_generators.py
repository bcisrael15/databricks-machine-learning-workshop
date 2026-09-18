# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic Data Generators
# MAGIC
# MAGIC Called by `01_setup.py`. Each function returns a pandas DataFrame.
# MAGIC All data is generated in-process — no external CSV dependencies.

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customers

# COMMAND ----------

def generate_customers(n=5000, seed=42):
    """Generate telecom customer profiles with realistic churn correlations (~33% churn rate).

    No third-party dependencies — names are drawn from small built-in lists so this runs
    on locked-down / air-gapped workspaces with no pip access. (The `name` column is
    cosmetic; it is never used as a model feature.)
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
        "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
        "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
        "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    ]

    contract_types = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    internet_services = ["Fiber optic", "DSL", "No"]

    records = []
    for i in range(n):
        customer_id = f"CUST-{i+1:05d}"
        name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
        gender = rng.choice(["Male", "Female"])
        senior_citizen = int(rng.random() < 0.16)
        age = int(rng.integers(65, 85)) if senior_citizen else int(rng.integers(18, 65))
        partner = rng.choice(["Yes", "No"])
        dependents = rng.choice(["Yes", "No"], p=[0.3, 0.7])

        # Contract type influences churn
        contract_type = rng.choice(contract_types, p=[0.50, 0.25, 0.25])
        tenure_months = int(
            rng.integers(1, 12) if contract_type == "Month-to-month"
            else rng.integers(6, 48) if contract_type == "One year"
            else rng.integers(12, 72)
        )

        internet_service = rng.choice(internet_services, p=[0.44, 0.34, 0.22])
        phone_service = rng.choice(["Yes", "No"], p=[0.9, 0.1])

        # Optional services (only if internet)
        has_internet = internet_service != "No"
        streaming_tv = rng.choice(["Yes", "No"]) if has_internet else "No internet service"
        streaming_movies = rng.choice(["Yes", "No"]) if has_internet else "No internet service"
        online_security = rng.choice(["Yes", "No"], p=[0.35, 0.65]) if has_internet else "No internet service"
        online_backup = rng.choice(["Yes", "No"], p=[0.4, 0.6]) if has_internet else "No internet service"
        device_protection = rng.choice(["Yes", "No"], p=[0.4, 0.6]) if has_internet else "No internet service"
        tech_support = rng.choice(["Yes", "No"], p=[0.35, 0.65]) if has_internet else "No internet service"

        paperless_billing = rng.choice(["Yes", "No"], p=[0.6, 0.4])
        payment_method = rng.choice(payment_methods)

        # Monthly charges based on services
        base_charge = 20.0
        if internet_service == "Fiber optic":
            base_charge += 44.0
        elif internet_service == "DSL":
            base_charge += 25.0
        if phone_service == "Yes":
            base_charge += 10.0
        for svc in [streaming_tv, streaming_movies, online_security, online_backup, device_protection, tech_support]:
            if svc == "Yes":
                base_charge += rng.uniform(5, 12)
        monthly_charges = round(base_charge + rng.uniform(-5, 10), 2)
        total_charges = round(monthly_charges * tenure_months + rng.uniform(-50, 50), 2)
        total_charges = max(total_charges, monthly_charges)

        # Churn probability: higher for month-to-month, short tenure, high charges, no support services
        churn_score = 0.0
        if contract_type == "Month-to-month":
            churn_score += 0.25
        elif contract_type == "One year":
            churn_score += 0.05
        if tenure_months < 6:
            churn_score += 0.15
        elif tenure_months < 12:
            churn_score += 0.08
        if monthly_charges > 80:
            churn_score += 0.10
        elif monthly_charges > 60:
            churn_score += 0.05
        if online_security == "No" and tech_support == "No":
            churn_score += 0.08
        if payment_method == "Electronic check":
            churn_score += 0.06
        if internet_service == "Fiber optic":
            churn_score += 0.05
        churn_score = min(churn_score, 0.85)
        churn = "Yes" if rng.random() < churn_score else "No"

        records.append({
            "customer_id": customer_id,
            "name": name,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "age": age,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "phone_service": phone_service,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "paperless_billing": paperless_billing,
            "churn": churn,
        })

    return pd.DataFrame(records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Service Tickets

# COMMAND ----------

def generate_service_tickets(customers_df, n=15000, seed=42):
    """Generate service tickets correlated with churn behavior."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    categories = ["billing", "technical", "cancellation", "upgrade", "general"]
    priorities = ["low", "medium", "high", "critical"]
    statuses = ["open", "resolved", "escalated"]

    churner_ids = set(customers_df[customers_df["churn"] == "Yes"]["customer_id"])
    all_ids = list(customers_df["customer_id"])

    # Churners get ~3x more tickets
    weights = customers_df["churn"].map({"Yes": 3.0, "No": 1.0}).values
    weights = weights / weights.sum()

    ticket_descriptions = {
        "billing": [
            "Customer inquired about unexpected charge on their latest bill.",
            "Dispute over billing amount — customer claims overcharge for current cycle.",
            "Customer requesting breakdown of charges on monthly statement.",
            "Payment processing error reported — double charge on credit card.",
            "Customer asking about proration after mid-cycle plan change.",
            "Late fee dispute — customer says payment was submitted on time.",
            "Customer confused about taxes and surcharges on bill.",
            "Request to change billing date to align with pay schedule.",
        ],
        "technical": [
            "Intermittent internet connectivity — customer reports frequent drops.",
            "Slow download speeds not matching subscribed plan tier.",
            "Router not connecting after power outage — needs reset instructions.",
            "Customer unable to stream video — buffering issues during peak hours.",
            "Wi-Fi signal weak in parts of the home — requesting range extender.",
            "Email service not syncing on mobile device.",
            "DNS resolution failures reported across multiple devices.",
            "VoIP call quality degradation — choppy audio and latency.",
        ],
        "cancellation": [
            "Customer wants to cancel service — moving to a new provider.",
            "Requesting cancellation due to pricing — found cheaper alternative.",
            "Customer frustrated with service quality and wants to terminate.",
            "Cancellation request — customer relocating out of service area.",
            "Customer threatening to cancel unless given a better rate.",
            "Account cancellation requested — switching to competitor.",
        ],
        "upgrade": [
            "Customer interested in upgrading to a faster internet tier.",
            "Inquiry about adding streaming package to current plan.",
            "Customer wants to upgrade from DSL to fiber optic service.",
            "Request to add international calling to phone plan.",
            "Customer exploring premium channel add-on options.",
            "Inquiry about bundle discount for adding mobile service.",
        ],
        "general": [
            "Customer requesting update to account contact information.",
            "General inquiry about contract renewal terms.",
            "Customer asking about available loyalty rewards.",
            "Request for copy of service agreement.",
            "Customer inquiring about referral program benefits.",
            "Question about service availability at a new address.",
        ],
    }

    records = []
    base_date = datetime(2025, 1, 1)
    for i in range(n):
        cust_idx = rng.choice(len(all_ids), p=weights)
        customer_id = all_ids[cust_idx]
        is_churner = customer_id in churner_ids

        # Churners bias toward cancellation/billing
        if is_churner:
            cat = rng.choice(categories, p=[0.30, 0.20, 0.25, 0.05, 0.20])
        else:
            cat = rng.choice(categories, p=[0.20, 0.30, 0.05, 0.20, 0.25])

        priority = rng.choice(priorities, p=[0.30, 0.40, 0.20, 0.10])
        if cat == "cancellation":
            priority = rng.choice(priorities, p=[0.05, 0.20, 0.45, 0.30])

        status = rng.choice(statuses, p=[0.15, 0.65, 0.20])
        resolution_hours = round(float(rng.exponential(24)) + 0.5, 1) if status == "resolved" else None
        created_date = base_date + timedelta(days=int(rng.integers(0, 365)), hours=int(rng.integers(8, 20)))
        description = rng.choice(ticket_descriptions[cat])

        records.append({
            "ticket_id": f"TKT-{i+1:06d}",
            "customer_id": customer_id,
            "created_date": created_date,
            "category": cat,
            "priority": priority,
            "status": status,
            "resolution_time_hours": resolution_hours,
            "description": description,
        })

    return pd.DataFrame(records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Churn Labels

# COMMAND ----------

def generate_churn_labels(customers_df, seed=42):
    """Extract churn labels into separate table with train/val/test split."""
    rng = np.random.default_rng(seed)

    labels = customers_df[["customer_id", "churn"]].copy()
    labels["label_date"] = datetime(2025, 6, 1)

    # 70/15/15 split
    n = len(labels)
    splits = rng.choice(["train", "val", "test"], size=n, p=[0.70, 0.15, 0.15])
    labels["split"] = splits

    return labels
