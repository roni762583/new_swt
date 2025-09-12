#!/bin/bash
# Setup GitHub repo while keeping large files locally for selective zipping

set -e

echo "🚀 Setting up GitHub repository with local large file management..."
echo ""

# 1. Create backup of large files
echo "📦 Step 1: Backing up large files locally..."
mkdir -p ../new_swt_large_files
cp -v checkpoints/*.pth ../new_swt_large_files/ 2>/dev/null || echo "  No .pth files to backup"
cp -v data/*.csv ../new_swt_large_files/ 2>/dev/null || echo "  No .csv files to backup"
echo "✅ Large files backed up to: ../new_swt_large_files/"
echo ""

# 2. Update .gitignore to exclude large files
echo "📝 Step 2: Updating .gitignore for large files..."
cat >> .gitignore << 'EOF'

# Large files (managed separately)
checkpoints/*.pth
checkpoints/*.pt
checkpoints/*.ckpt
data/*.csv
data/*.parquet
*.zip
*.tar.gz
*.7z

# But track README files in these directories
!checkpoints/README.md
!data/README.md
EOF
echo "✅ .gitignore updated"
echo ""

# 3. Create README files for large file directories
echo "📄 Step 3: Creating placeholder READMEs..."

cat > checkpoints/README.md << 'EOF'
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
EOF

cat > data/README.md << 'EOF'
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
EOF

echo "✅ README files created"
echo ""

# 4. Create snapshot script
echo "🔧 Step 4: Creating snapshot script..."

cat > create_snapshot.sh << 'EOF'
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
EOF

chmod +x create_snapshot.sh
echo "✅ Snapshot script created: ./create_snapshot.sh"
echo ""

# 5. Create restore script
echo "🔧 Step 5: Creating restore script..."

cat > restore_large_files.sh << 'EOF'
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
EOF

chmod +x restore_large_files.sh
echo "✅ Restore script created: ./restore_large_files.sh"
echo ""

# 6. Initialize Git repository
echo "📦 Step 6: Initializing Git repository..."
git init
git add -A
git status --short
echo ""

# 7. Show what will be committed
echo "📊 Repository Statistics:"
echo "  Files to commit: $(git ls-files | wc -l)"
echo "  Repository size (without large files): $(du -sh --exclude=.git --exclude='*.pth' --exclude='*.csv' . | cut -f1)"
echo "  Large files (stored locally):"
ls -lh ../new_swt_large_files/ 2>/dev/null || echo "    None yet"
echo ""

# 8. Create initial commit
echo "💾 Step 7: Creating initial commit..."
git commit -m "Initial commit: SWT Production Trading System

- Complete training and validation framework
- Docker deployment configurations
- OANDA integration
- Large files (checkpoints, data) managed locally
- Use create_snapshot.sh to create full archives"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Local repository ready! Next steps:"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1. Create private repository on GitHub:"
echo "   https://github.com/new"
echo "   - Name: new_swt"
echo "   - Private: YES"
echo "   - Initialize: NO (no README, .gitignore, or license)"
echo ""
echo "2. Push to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/new_swt.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Workflow for snapshots:"
echo "   ./create_snapshot.sh v1.0.0  # Creates full archive with large files"
echo "   ./restore_large_files.sh     # Restores large files for local work"
echo ""
echo "📁 Directory Structure:"
echo "   new_swt/                 (GitHub repo - no large files)"
echo "   new_swt_large_files/     (Local backup of large files)"
echo "   new_swt_snapshots/       (Full archives at key commits)"
echo ""