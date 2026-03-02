"""
benchmark/fixtures/generate_fixtures.py

Generates all fixture files used by the benchmark task suite.
Run once before benchmarking:
    python benchmark/fixtures/generate_fixtures.py
"""

import os
import json
import random
import pandas as pd

OUT = os.path.dirname(__file__)
random.seed(42)


# ── people.csv ──────────────────────────────────────────────────────────
# Used by: adversarial_planner_normalization_test
people = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"],
    "age":    [29, 22, 35, 41, 25, 33, 28, 45, 19, 37],
    "city":   ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago"],
    "region": ["East", "West", "East", "Central", "West", "East", "Central", "West", "East", "Central"],
    "salary": [55000, 38000, 72000, 91000, 42000, 67000, 51000, 88000, 31000, 74000],
})
people.loc[4, "salary"] = None
people.to_csv(os.path.join(OUT, "people.csv"), index=False)
print(f"Generated people.csv ({len(people)} rows)")


# ── employees.csv ───────────────────────────────────────────────────────
# Used by: etl_multi_stage_pipeline, csv_aggregate_export, csv_full_pipeline
# Added region column — required by etl_multi_stage_pipeline Aggregator
departments = ["Engineering", "Marketing", "Sales", "HR", "Engineering", "Sales"]
regions = ["East", "West", "Central"]
employees = pd.DataFrame({
    "name":       [f"Employee_{i}" for i in range(20)],
    "department": [departments[i % len(departments)] for i in range(20)],
    "region":     [regions[i % len(regions)] for i in range(20)],
    "salary":     [random.randint(30000, 100000) for _ in range(20)],
    "tenure":     [random.randint(1, 10) for _ in range(20)],
})
employees.loc[3, "salary"] = None
duplicates = employees.iloc[:3].copy()
employees = pd.concat([employees, duplicates], ignore_index=True)
employees.to_csv(os.path.join(OUT, "employees.csv"), index=False)
print(f"Generated employees.csv ({len(employees)} rows, includes nulls and duplicates)")


# ── customers.csv ───────────────────────────────────────────────────────
# Used by: postgres_pipeline_with_logging
customers = pd.DataFrame({
    "name":        [f"Customer_{i}" for i in range(20)],
    "status":      [random.choice(["active", "active", "inactive"]) for _ in range(20)],
    "signup_date": pd.date_range("2023-01-01", periods=20, freq="ME").strftime("%Y-%m-%d").tolist(),
    "region":      [random.choice(["East", "West", "Central"]) for _ in range(20)],
})
customers.to_csv(os.path.join(OUT, "customers.csv"), index=False)
print(f"Generated customers.csv ({len(customers)} rows)")


# ── predictions.csv ─────────────────────────────────────────────────────
# Used by: rest_api_sqlite_pipeline
predictions = pd.DataFrame({
    "name":  [f"Item_{i}" for i in range(30)],
    "score": [round(random.uniform(0.0, 1.0), 3) for _ in range(30)],
    "label": [random.choice(["A", "B", "C"]) for _ in range(30)],
})
predictions.loc[5, "score"] = None
predictions.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
print(f"Generated predictions.csv ({len(predictions)} rows)")


# ── events.json ─────────────────────────────────────────────────────────
# Used by: json_ingest_sql_query_export
event_types = ["click", "view", "purchase", "signup", "logout"]
events = [
    {
        "event_id":   i,
        "event_type": random.choice(event_types),
        "user_id":    random.randint(1000, 1099),
        "timestamp":  f"2024-0{random.randint(1,9)}-{random.randint(10,28)}T{random.randint(0,23):02d}:00:00Z",
        "value":      round(random.uniform(0, 500), 2)
    }
    for i in range(50)
]
with open(os.path.join(OUT, "events.json"), "w") as f:
    json.dump(events, f, indent=2)
print(f"Generated events.json ({len(events)} records)")


print("\nAll fixtures ready in benchmark/fixtures/")