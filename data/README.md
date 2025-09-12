# Data Files

Large data files are managed locally and not stored in Git.

## GBPJPY M1 Data
- **File:** `GBPJPY_M1_202201-202508.csv`
- **Size:** 6.3MB
- **Period:** 2022-01 to 2025-08
- **Rows:** 63,361

## To restore data:
```bash
# From backup
cp ../new_swt_large_files/*.csv data/

# Or extract from snapshot
tar -xzf ../new_swt_snapshots/snapshot_v1.0.0.tar.gz --strip-components=1 data/
```
