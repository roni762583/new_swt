#!/bin/bash
# Speed Demon GCP Setup - n2d-standard-224 (224 cores!)
# This script deploys your MuZero training to a massive cloud instance

set -e  # Exit on error

# Configuration
PROJECT_ID="alpine-task-472915-h0"  # CHANGE THIS to your GCP project
INSTANCE_NAME="muzero-monster"
ZONE="us-central1-a"
MACHINE_TYPE="n2d-standard-224"
DISK_SIZE="200"  # GB

echo "🚀 SPEED DEMON DEPLOYMENT SCRIPT"
echo "================================"
echo "Machine: $MACHINE_TYPE (224 cores)"
echo "Cost: ~$2.00/hour (spot pricing)"
echo "Speed: ~5,000 episodes/hour"
echo ""

# Step 1: Create the monster instance
echo "📦 Step 1: Creating GCP instance..."
gcloud compute instances create $INSTANCE_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=$DISK_SIZE \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --metadata startup-script='#!/bin/bash
    # Install Docker
    apt-get update
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io

    # Install Docker Compose
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose

    # Optimize for maximum CPU performance
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

    # Increase file limits
    echo "* soft nofile 65536" >> /etc/security/limits.conf
    echo "* hard nofile 65536" >> /etc/security/limits.conf

    # Docker daemon optimization
    cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2"
}
EOF
    systemctl restart docker
    '

echo "⏳ Waiting for instance to be ready..."
sleep 30

# Step 2: Create optimized docker-compose for 224 cores
echo "📝 Step 2: Creating optimized docker-compose..."
cat > /tmp/docker-compose.cloud.yml <<'EOF'
version: '3.8'

services:
  micro-training:
    image: micro-muzero:latest
    container_name: micro_training_cloud
    restart: unless-stopped
    environment:
      - PYTHONPATH=/workspace
      - OMP_NUM_THREADS=4  # Each worker gets 4 threads
      - MKL_NUM_THREADS=4
      - NUMBA_NUM_THREADS=4
      - NUM_WORKERS=50  # Increased for 224 cores
    volumes:
      - ./data:/workspace/data:ro
      - ./micro:/workspace/micro
      - ./micro/checkpoints:/workspace/micro/checkpoints
      - ./micro/logs:/workspace/micro/logs
    command: python3 -u /workspace/micro/training/train_micro_muzero.py
    deploy:
      resources:
        limits:
          memory: 400G
          cpus: '220'  # Leave 4 for system
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  micro-validation:
    image: micro-muzero:latest
    container_name: micro_validation_cloud
    restart: unless-stopped
    environment:
      - PYTHONPATH=/workspace
      - MC_RUNS=200
      - VALIDATION_TIMEOUT=3600
    volumes:
      - ./data:/workspace/data:ro
      - ./micro:/workspace/micro
      - ./micro/checkpoints:/workspace/micro/checkpoints
      - ./micro/validation_results:/workspace/micro/validation_results
    command: python3 /workspace/micro/validation/validate_micro_watcher.py
    deploy:
      resources:
        limits:
          memory: 50G
          cpus: '4'
EOF

# Step 3: Create updated training config for 224 cores
echo "⚙️ Step 3: Creating optimized training config..."
cat > /tmp/train_config_cloud.py <<'EOF'
# Add this to your train_micro_muzero.py or override TrainingConfig

# Optimized for 224 cores
NUM_WORKERS = 50  # Massively parallel episode collection
BATCH_SIZE = 512  # Larger batch with more RAM
EPISODES_PER_ITERATION = 10  # Collect more episodes per iteration

print(f"Cloud config: {NUM_WORKERS} workers on 224 cores")
print(f"Expected speed: ~5,000 episodes/hour")
EOF

# Step 4: Package current state
echo "📦 Step 4: Packaging your current training state..."
cd /home/aharon/projects/new_swt

# Save current Docker image
echo "Saving Docker image..."
docker save micro-muzero:latest | gzip > /tmp/micro-muzero.tar.gz

# Create deployment package
tar -czf /tmp/muzero-deploy.tar.gz \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  micro/ data/ docker-compose.yml

echo "📤 Step 5: Uploading to cloud instance..."
# Upload Docker image
gcloud compute scp /tmp/micro-muzero.tar.gz $INSTANCE_NAME:/tmp/ --zone=$ZONE

# Upload code and data
gcloud compute scp /tmp/muzero-deploy.tar.gz $INSTANCE_NAME:/tmp/ --zone=$ZONE

# Upload optimized configs
gcloud compute scp /tmp/docker-compose.cloud.yml $INSTANCE_NAME:~/docker-compose.yml --zone=$ZONE
gcloud compute scp /tmp/train_config_cloud.py $INSTANCE_NAME:/tmp/ --zone=$ZONE

# Step 6: Setup and start on cloud
echo "🔧 Step 6: Setting up on cloud instance..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='
  cd ~
  echo "Extracting deployment package..."
  tar -xzf /tmp/muzero-deploy.tar.gz

  echo "Loading Docker image..."
  docker load < /tmp/micro-muzero.tar.gz

  echo "Updating training config for 224 cores..."
  sed -i "s/num_workers: int = .*/num_workers: int = 50/" micro/training/train_micro_muzero.py
  sed -i "s/batch_size: int = .*/batch_size: int = 512/" micro/training/train_micro_muzero.py
  sed -i "s/episodes_per_iteration: int = .*/episodes_per_iteration: int = 10/" micro/training/train_micro_muzero.py

  echo "Starting training..."
  docker-compose up -d

  echo "Training started! Checking status..."
  sleep 10
  docker ps
  docker logs micro_training_cloud --tail 20
'

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo "======================="
echo ""
echo "📊 Monitoring Commands:"
echo "  SSH to instance:  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "  View logs:        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- docker logs -f micro_training_cloud"
echo "  Check progress:   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- 'docker logs micro_training_cloud 2>&1 | grep Episode | tail'"
echo ""
echo "💰 Cost Management:"
echo "  Stop instance:    gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo "  Start instance:   gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
echo "  Delete instance:  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "📈 Expected Performance:"
echo "  Speed: ~5,000 episodes/hour"
echo "  Episode 10,000: ~2 hours (~\$4)"
echo "  Episode 50,000: ~10 hours (~\$20)"
echo "  Episode 100,000: ~20 hours (~\$40)"
echo ""
echo "⚠️ REMEMBER: This is a SPOT instance - it may be preempted. Checkpoints are saved every 50 episodes."