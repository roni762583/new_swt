# Checkpoint Files

Large checkpoint files are managed locally and not stored in Git.

## Episode 13475 Checkpoint
- **File:** `episode_13475.pth`
- **Size:** 439MB
- **Purpose:** Production trading model
- **Created:** September 2024

## To restore checkpoints:
```bash
# From backup
cp ../new_swt_large_files/*.pth checkpoints/

# Or extract from snapshot
tar -xzf ../new_swt_snapshots/snapshot_v1.0.0.tar.gz --strip-components=1 checkpoints/
```
