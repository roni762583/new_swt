#!/bin/bash
# Create a snapshot of the entire project including large files

VERSION=${1:-$(git describe --tags --always 2>/dev/null || echo "uncommitted")}
DATE=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_NAME="new_swt_snapshot_${VERSION}_${DATE}"
SNAPSHOT_DIR="../new_swt_snapshots"

echo "📸 Creating snapshot: $SNAPSHOT_NAME"

# Create snapshots directory
mkdir -p "$SNAPSHOT_DIR"

# Copy large files back temporarily
echo "  Restoring large files..."
cp ../new_swt_large_files/*.pth checkpoints/ 2>/dev/null || true
cp ../new_swt_large_files/*.csv data/ 2>/dev/null || true

# Create compressed archive
echo "  Creating archive..."
tar -czf "$SNAPSHOT_DIR/${SNAPSHOT_NAME}.tar.gz" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env.local' \
  --exclude='logs/*' \
  --exclude='validation_results/*' \
  --exclude='test_results/*' \
  .

# Remove large files again
echo "  Cleaning up large files..."
rm -f checkpoints/*.pth
rm -f data/*.csv

# Create checksum
echo "  Generating checksum..."
cd "$SNAPSHOT_DIR"
md5sum "${SNAPSHOT_NAME}.tar.gz" > "${SNAPSHOT_NAME}.md5"
SIZE=$(du -h "${SNAPSHOT_NAME}.tar.gz" | cut -f1)

echo ""
echo "✅ Snapshot created:"
echo "   File: $SNAPSHOT_DIR/${SNAPSHOT_NAME}.tar.gz"
echo "   Size: $SIZE"
echo "   MD5: $(cat ${SNAPSHOT_NAME}.md5)"
echo ""
echo "To restore: tar -xzf ${SNAPSHOT_NAME}.tar.gz"
