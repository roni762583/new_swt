#!/bin/bash
# Entrypoint script - use master.duckdb directly (no precomputing needed)

# ❌ DEPRECATED: precompute_features_to_db.py (aggregates M1→M5, loses granularity)
# ✅ CURRENT: Use master.duckdb with M1 data directly

# Check if master database exists
if [ -f "/app/master.duckdb" ]; then
    echo "✅ Using master.duckdb (M1 data with precomputed features)"
elif [ -f "/app/data/master.duckdb" ]; then
    echo "✅ Using /app/data/master.duckdb (M1 data with precomputed features)"
else
    echo "❌ ERROR: master.duckdb not found!"
    echo "Expected locations:"
    echo "  - /app/master.duckdb (mounted via docker-compose)"
    echo "  - /app/data/master.duckdb"
    exit 1
fi

# Execute the main command
exec "$@"