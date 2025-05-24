import os
import subprocess

print("🚀 Starting Predictive Maintenance Pipeline...\n")

# 1. Feature engineering
print("🔧 Step 1: Running feature engineering...")
subprocess.run(["python", "scripts/feature_engineering.py"])

# 2. Model training
print("\n🧠 Step 2: Training models...")
subprocess.run(["python", "scripts/train.py"])

# 3. Evaluation
print("\n📊 Step 3: Evaluating models...")
subprocess.run(["python", "scripts/evaluate.py"])

print("\n✅ Pipeline complete! Results are in the 'output/' and 'models/' folders.")
