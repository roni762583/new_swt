#!/bin/bash
# Restore large files from backup

echo "🔄 Restoring large files..."

if [ -d "../new_swt_large_files" ]; then
    cp -v ../new_swt_large_files/*.pth checkpoints/ 2>/dev/null || echo "No .pth files found"
    cp -v ../new_swt_large_files/*.csv data/ 2>/dev/null || echo "No .csv files found"
    echo "✅ Large files restored"
else
    echo "❌ Backup directory not found: ../new_swt_large_files"
    echo "   You may need to extract from a snapshot instead"
fi
