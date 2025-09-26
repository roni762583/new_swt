#!/usr/bin/env python3
import duckdb

# Connect to database
conn = duckdb.connect('/data/micro_features.duckdb', read_only=True)

# Get column names
result = conn.execute("SELECT * FROM micro_features LIMIT 1").description
columns = [col[0] for col in result]

print("Available columns:")
print("=" * 50)
for i, col in enumerate(columns):
    print(f"{i:3d}: {col}")

# Filter for base features (lag 0)
print("\nBase features (lag 0):")
print("=" * 50)
base_cols = [col for col in columns if col.endswith('_0') or col in ['bar_index', 'price_change_pips']]
for col in base_cols:
    print(col)