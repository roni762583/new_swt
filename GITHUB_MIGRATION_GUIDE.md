# 🚀 GitHub Migration Guide for new_swt

## Large File Handling Strategy

### Option 1: Git LFS (Recommended for Private Repos)
**Pros:** Keep everything in one place, automatic versioning
**Cons:** Costs money for large files (439MB checkpoint)

```bash
# Run the automated script
chmod +x migrate_to_github.sh
./migrate_to_github.sh
```

**Git LFS Pricing:**
- Free: 1GB storage, 1GB bandwidth/month
- Your checkpoint (439MB) fits in free tier
- Each clone/pull uses bandwidth
- $5/month for 50GB storage + 50GB bandwidth

---

### Option 2: Exclude Large Files (Most Economical)
**Pros:** Completely free, no LFS needed
**Cons:** Need to manually manage checkpoint files

```bash
# Initialize without large files
git init

# Create .gitignore for large files
cat >> .gitignore << 'EOF'
# Large model files (manage separately)
checkpoints/*.pth
checkpoints/*.pt
data/*.csv
data/*.parquet
EOF

# Add placeholder README for large files
cat > checkpoints/README.md << 'EOF'
# Checkpoint Files

Large checkpoint files are not stored in Git.

## Episode 13475 Checkpoint
- File: episode_13475.pth
- Size: 439MB
- MD5: [generate with: md5sum episode_13475.pth]

Download from: [Your cloud storage link]
EOF

# Commit without large files
git add -A
git commit -m "Initial commit (large files excluded)"

# Push to GitHub (no LFS needed)
git remote add origin https://github.com/YOUR_USERNAME/new_swt.git
git branch -M main
git push -u origin main
```

**Then store large files in:**
- Google Drive
- Dropbox
- AWS S3
- Azure Blob Storage
- Or share via secure link

---

### Option 3: Split Repository
**Pros:** Clean separation, no LFS costs
**Cons:** Two repos to manage

```bash
# Main repo (code only)
cd new_swt
git init
git add -A
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/new_swt.git
git push -u origin main

# Data repo (with LFS or releases)
# Create new_swt-models repo on GitHub
mkdir ../new_swt-models
cp checkpoints/*.pth ../new_swt-models/
cp data/*.csv ../new_swt-models/
cd ../new_swt-models
git init
git lfs track "*.pth" "*.csv"
git add -A
git commit -m "Model checkpoints and data"
git remote add origin https://github.com/YOUR_USERNAME/new_swt-models.git
git push -u origin main
```

---

### Option 4: GitHub Releases (Best for Checkpoints)
**Pros:** Free, designed for binary files, versioned
**Cons:** Manual upload process

```bash
# 1. Push code without large files (like Option 2)
git init
git add -A
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/new_swt.git
git push -u origin main

# 2. Create a release on GitHub
# Go to: https://github.com/YOUR_USERNAME/new_swt/releases/new
# - Tag version: v1.0.0
# - Release title: "Episode 13475 Checkpoint"
# - Attach binary: episode_13475.pth (439MB)
# - Attach binary: GBPJPY_M1_202201-202508.csv (6.3MB)

# 3. Users download with:
wget https://github.com/YOUR_USERNAME/new_swt/releases/download/v1.0.0/episode_13475.pth
wget https://github.com/YOUR_USERNAME/new_swt/releases/download/v1.0.0/GBPJPY_M1_202201-202508.csv
```

---

## 🎯 Recommendation

For your use case with a 439MB checkpoint:

1. **If solo/small team:** Use **Option 4 (GitHub Releases)** - It's free and designed for this
2. **If frequent updates:** Use **Option 1 (Git LFS)** - Worth the $5/month
3. **If cost-sensitive:** Use **Option 2 (Exclude)** - Store checkpoint in cloud storage

---

## Quick Start Commands

### For GitHub Releases (Recommended):
```bash
# Initialize and push code
cd /home/aharon/projects/new_muzero/new_swt
git init
git add -A
git commit -m "Initial commit: SWT Production Trading System"
git remote add origin https://github.com/YOUR_USERNAME/new_swt.git
git branch -M main
git push -u origin main

# Then manually upload checkpoint as a release
```

### Check file sizes:
```bash
du -h checkpoints/episode_13475.pth  # 439MB
du -h data/GBPJPY_M1_202201-202508.csv  # 6.3MB
du -sh .  # Total size
```

---

## Security Notes

⚠️ **NEVER commit .env file with real credentials!**

The `.gitignore` already excludes:
- `.env` (credentials)
- `*.pth` (if using Option 2)
- `*.csv` (if using Option 2)

Always verify before pushing:
```bash
git status  # Check what will be committed
git ls-files | grep -E "\.(env|pth|csv)$"  # Should be empty
```