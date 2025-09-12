#!/bin/bash
# Migration script for new_swt to GitHub with Git LFS for large files

set -e  # Exit on error

echo "🚀 Starting migration to GitHub..."

# 1. Initialize Git repository
echo "📦 Initializing Git repository..."
git init

# 2. Install and setup Git LFS for large files
echo "📂 Setting up Git LFS for large files..."
git lfs install

# Track large files with Git LFS
echo "🔍 Tracking large files with Git LFS..."
git lfs track "*.pth"  # Model checkpoints
git lfs track "*.pt"   # PyTorch files
git lfs track "*.ckpt" # Checkpoint files
git lfs track "*.h5"   # HDF5 files
git lfs track "*.pkl"  # Large pickle files
git lfs track "*.parquet" # Parquet files
git lfs track "data/*.csv" # Large CSV files

# This creates/updates .gitattributes
echo "✅ Git LFS configured"

# 3. Add all files
echo "📝 Adding files to repository..."
git add .gitattributes  # Add LFS tracking first
git add .gitignore      # Add ignore rules
git add -A              # Add everything else

# 4. Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: SWT Production Trading System

- Episode 13475 checkpoint included via Git LFS
- Complete validation framework
- Docker deployment ready
- OANDA integration configured"

# 5. Create GitHub repository instructions
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Local repository ready! Now create a private GitHub repository:"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "1. Go to https://github.com/new"
echo "2. Create a PRIVATE repository named 'new_swt'"
echo "3. DO NOT initialize with README, .gitignore, or license"
echo "4. After creating, run these commands:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/new_swt.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "📊 Repository Stats:"
echo "   - Checkpoint size: $(du -h checkpoints/episode_13475.pth 2>/dev/null | cut -f1)"
echo "   - Data size: $(du -h data/GBPJPY_M1_202201-202508.csv 2>/dev/null | cut -f1)"
echo "   - Total size: $(du -sh . | cut -f1)"
echo ""
echo "⚠️  IMPORTANT: GitHub has limits for Git LFS:"
echo "   - Free tier: 1GB storage, 1GB bandwidth/month"
echo "   - Large files like episode_13475.pth (439MB) will use LFS quota"
echo "   - Consider GitHub Pro for 2GB storage if needed"
echo ""