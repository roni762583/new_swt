#!/bin/bash
# Entrypoint script to precompute features if needed

# Check if precomputed features database exists
if [ ! -f "/app/precomputed_features.duckdb" ]; then
    echo "🔄 Precomputed features not found, generating..."

    # Check if master database exists
    if [ -f "/app/data/master.duckdb" ]; then
        echo "📊 Found master database, computing features..."
        python3 precompute_features_to_db.py

        if [ $? -eq 0 ]; then
            echo "✅ Features precomputed successfully"
        else
            echo "❌ Failed to precompute features, creating sample data..."
            # The script will create sample data if master.duckdb doesn't exist
            python3 precompute_features_to_db.py
        fi
    else
        echo "⚠️ Master database not found at /app/data/master.duckdb"
        echo "📦 Creating sample data for testing..."
        python3 precompute_features_to_db.py
    fi
else
    echo "✅ Precomputed features already exist"
fi

# Execute the main command
exec "$@"