#!/bin/bash
# Clean up old Docker files and keep only unified versions

echo "🧹 Cleaning up old Docker files..."

# List of old files to remove
OLD_FILES=(
    "Dockerfile"
    "Dockerfile.minimal"
    "Dockerfile.buildkit"
    "Dockerfile.optimized"
    "docker-compose.yml"
    "docker-compose.minimal.yml"
    "docker-compose.buildkit.yml"
    "requirements-minimal.txt"
    "requirements-optimized.txt"
)

# Backup old files
mkdir -p .old_docker_backup
for file in "${OLD_FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" .old_docker_backup/
        echo "  Moved $file to backup"
    fi
done

# Rename unified files to standard names
if [ -f "Dockerfile.unified" ]; then
    mv Dockerfile.unified Dockerfile
    echo "✅ Renamed Dockerfile.unified → Dockerfile"
fi

if [ -f "docker-compose.unified.yml" ]; then
    mv docker-compose.unified.yml docker-compose.yml
    echo "✅ Renamed docker-compose.unified.yml → docker-compose.yml"
fi

echo "✨ Cleanup complete! Old files backed up to .old_docker_backup/"
echo ""
echo "📋 Next steps:"
echo "  1. Build: DOCKER_BUILDKIT=1 docker compose build"
echo "  2. Run:   docker compose up -d"
echo "  3. Logs:  docker compose logs -f ppo-training"
