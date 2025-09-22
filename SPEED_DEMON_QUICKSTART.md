# 🚀 SPEED DEMON QUICK START
## Get to Episode 100,000 in 20 hours for $40

### Prerequisites
1. **GCP Account** with billing enabled
2. **gcloud CLI** installed and authenticated
3. **Current training** at episode 5,600+

---

## ⚡ QUICK SETUP (10 minutes)

### 1️⃣ Set Your Project ID
```bash
# First, find your project ID
gcloud projects list

# Set it in the script
sed -i 's/your-project-id/YOUR_ACTUAL_PROJECT_ID/' deploy_speed_demon.sh
```

### 2️⃣ Make Scripts Executable
```bash
chmod +x deploy_speed_demon.sh
chmod +x monitor_cloud.sh
```

### 3️⃣ Deploy to Cloud (One Command!)
```bash
./deploy_speed_demon.sh
```
This will:
- Create a 224-core monster instance
- Upload your current training state
- Start training at ~5,000 episodes/hour

### 4️⃣ Monitor Progress
```bash
# In a new terminal
./monitor_cloud.sh
```

---

## 📊 WHAT YOU'LL SEE

### Speed Comparison
| Location | Episodes/Hour | Time to 100k | Cost |
|----------|--------------|--------------|------|
| Your Laptop | 130 | 770 hours (32 days) | $0 |
| **Speed Demon** | **5,000** | **20 hours** | **$40** |

### Expected Timeline
- **2 hours**: Episode 15,000 (statistical significance!)
- **4 hours**: Episode 25,000 (robust confidence)
- **10 hours**: Episode 55,000 (professional level)
- **20 hours**: Episode 105,000 (production ready)

---

## 🎮 ESSENTIAL COMMANDS

### Check Training Progress
```bash
# See latest episodes
gcloud compute ssh muzero-monster --zone=us-central1-a -- \
  'docker logs micro_training_cloud 2>&1 | grep Episode | tail -5'

# Watch live logs
gcloud compute ssh muzero-monster --zone=us-central1-a -- \
  'docker logs -f micro_training_cloud'
```

### Download Checkpoints
```bash
# Get the best checkpoint
gcloud compute scp \
  muzero-monster:~/micro/checkpoints/best.pth \
  ./checkpoints/best_cloud.pth \
  --zone=us-central1-a
```

### Cost Management
```bash
# Stop (keeps data, stops billing)
gcloud compute instances stop muzero-monster --zone=us-central1-a

# Resume
gcloud compute instances start muzero-monster --zone=us-central1-a

# Delete (removes everything)
gcloud compute instances delete muzero-monster --zone=us-central1-a
```

---

## ⚠️ IMPORTANT NOTES

### Spot Instance Preemption
- Your instance may be stopped if GCP needs capacity
- Checkpoints save every 50 episodes (safe)
- Just restart if preempted: `gcloud compute instances start ...`

### First Hour Checklist
- [ ] Instance created successfully
- [ ] Docker containers running
- [ ] Episodes incrementing (check: ~5000/hr)
- [ ] Checkpoints being saved
- [ ] Cost tracking working

---

## 🚨 TROUBLESHOOTING

### "Permission Denied"
```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable Compute API
gcloud services enable compute.googleapis.com
```

### "Quota Exceeded"
```bash
# Request quota increase for N2D CPUs
# Go to: https://console.cloud.google.com/iam-admin/quotas
# Search: "N2D CPUs"
# Request: 250 (for 224 + buffer)
```

### Container Not Starting
```bash
# SSH and check
gcloud compute ssh muzero-monster --zone=us-central1-a

# Inside instance:
docker ps -a
docker logs micro_training_cloud
```

---

## 💰 BUDGET ALERTS

Set up billing alert:
```bash
# In GCP Console: Billing → Budgets & Alerts
# Set alert at $50 (safe margin for 24 hours)
```

---

## 🎯 SUCCESS METRICS

You'll know it's working when:
1. **Episode rate**: 4,000-5,000 per hour
2. **CPU usage**: 90%+ across 224 cores
3. **Expectancy**: Improving every 1,000 episodes
4. **Cost**: ~$2/hour accumulating

---

## 🏁 QUICK START COMMAND

**Copy-paste this to start immediately:**
```bash
# One-liner to rule them all
curl -sSL https://raw.githubusercontent.com/your-repo/deploy_speed_demon.sh | \
  sed "s/your-project-id/$(gcloud config get-value project)/" | \
  bash
```

**That's it! You'll be training at 5,000 episodes/hour in 10 minutes!** 🚀