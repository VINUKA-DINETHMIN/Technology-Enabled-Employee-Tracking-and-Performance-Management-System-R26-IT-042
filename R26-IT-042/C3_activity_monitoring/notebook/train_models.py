# --- ADDED: config + safe loader (PLACE THIS BEFORE any other imports) ---
import os
import sys
import time
from pathlib import Path

# Try multiple candidate data directories (normal + editor-style with colon)
CANDIDATE_DATA_DIRS = [
    "/Users/vinukadinethmin/Desktop/C3_activity_monitoring/data",
    "/Users/vinukadinethmin/Desktop/C3_activity_monitoring:/data",
]

DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if Path(d).exists():
        DATA_DIR = d
        break
# fallback to first candidate if none found (safe_read_csv will handle missing file)
if DATA_DIR is None:
    DATA_DIR = CANDIDATE_DATA_DIRS[0]

DATASET_PATH = os.path.join(DATA_DIR, "employee_monitoring_dataset.csv")

def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

_log(f"CONFIG: DATA_DIR = {DATA_DIR}")
_log(f"CONFIG: DATASET_PATH = {DATASET_PATH}")


def safe_read_csv(path, **kwargs):
    """Load CSV with clear messages and helpful error output if file missing."""
    _log("Loading dataset...")
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        _log("ERROR: Dataset CSV not found.")
        print(f"  Expected path: {path}")
        try:
            files = sorted(os.listdir(DATA_DIR))
            print(f"  Files in data folder ({DATA_DIR}):")
            for f in files:
                print(f"    - {f}")
        except Exception as e:
            print(f"  Could not list data directory {DATA_DIR}: {e}")
        # stop execution so user can fix the path
        sys.exit(1)
    # lazy import to avoid requiring pandas until needed
    import pandas as _pd
    df = _pd.read_csv(path, **kwargs)
    _log(f"Loaded dataset ({len(df):,} rows, {len(df.columns):,} cols)")
    return df

# Ensure # %% markers remain harmless when run as a script (they're comments)
# --- END ADDED BLOCK ---

# %% [markdown]
# # R26-IT-042 — Anomaly Detection Model Training
# **Component C3: Activity Monitoring — R.K. Vinuka Dinethmin (IT22248642)**
#
# This notebook trains both models using your dataset.
# Run each cell one by one in VS Code with the Jupyter extension.

# %% [markdown]
# ## Cell 1 — Install dependencies

# %%
# Run this cell first to install everything needed
# In VS Code terminal: pip install scikit-learn pandas numpy matplotlib seaborn joblib

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "joblib", "-q"])
print("All packages installed.")

# %% [markdown]
# ## Cell 2 — Imports and config

# %%
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")
print("Imports OK")

FEATURE_COLS = [
    'mean_dwell_time', 'std_dwell_time', 'mean_flight_time',
    'typing_speed_wpm', 'error_rate',
    'mean_velocity', 'std_velocity', 'mean_acceleration',
    'mean_curvature', 'click_frequency', 'idle_ratio',
    'app_switch_frequency', 'active_app_entropy', 'total_focus_duration',
    'session_duration_min', 'geolocation_deviation',
    'wifi_ssid_match', 'device_fingerprint_match', 'face_liveness_score'
]
print(f"Feature columns: {len(FEATURE_COLS)}")

# %% [markdown]
# ## Cell 3 — Load dataset

# %%
# Use the absolute DATASET_PATH defined at the top of the file
# (the top-level block defines DATASET_PATH already)
# Remove any duplicate manual DATASET_PATH assignment below
# and use the safe reader that prints helpful errors.
df = safe_read_csv(DATASET_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nFirst 3 rows:")
print(df[FEATURE_COLS + ['label']].head(3).to_string(index=False))

# %% [markdown]
# ## Cell 4 — Data quality check

# %%
print("=== Data Quality Check ===\n")

# Missing values
nulls = df[FEATURE_COLS].isnull().sum()
print(f"Missing values: {nulls.sum()} (should be 0)")

# Feature statistics
print(f"\nFeature statistics:")
print(df[FEATURE_COLS].describe().round(3).to_string())

# Class balance
normal_count   = (df['label'] == 'normal').sum()
anomaly_count  = (df['label'] != 'normal').sum()
print(f"\nNormal samples  : {normal_count:,} ({normal_count/len(df)*100:.1f}%)")
print(f"Anomaly samples : {anomaly_count:,} ({anomaly_count/len(df)*100:.1f}%)")

# %% [markdown]
# ## Cell 5 — Train/Test split

# %%
# 80% train, 20% test — stratified to keep label balance
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

print(f"Train set : {len(train_df):,} rows")
print(f"Test set  : {len(test_df):,} rows")
print(f"\nTrain label counts:")
print(train_df['label'].value_counts())
print(f"\nTest label counts:")
print(test_df['label'].value_counts())

# %% [markdown]
# ## Cell 6 — Train Isolation Forest (PRIMARY MODEL)

# %%
_log('=' * 50)
_log('Training Isolation Forest (Primary Model)')
_log('=' * 50)

# Train ONLY on normal data
normal_train = train_df[train_df['label'] == 'normal']
_log(f"Training on {len(normal_train):,} normal samples...")

# Scale features
if_scaler  = StandardScaler()
X_normal   = normal_train[FEATURE_COLS].values
X_scaled   = if_scaler.fit_transform(X_normal)

# Train model
_log('Training Isolation Forest model...')
if_model = IsolationForest(
    contamination=0.05,
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
if_model.fit(X_scaled)

_log('Training complete!')

# Save models
_log('Saving models to ../models/...')
os.makedirs("../models", exist_ok=True)
with open("../models/if_model.pkl", "wb") as f:
    pickle.dump(if_model, f)
with open("../models/if_scaler.pkl", "wb") as f:
    pickle.dump(if_scaler, f)

_log('Models saved to ../models/')

# %% [markdown]
# ## Cell 7 — Evaluate Isolation Forest

# %%
print("=" * 50)
print("  Isolation Forest — Evaluation")
print("=" * 50)

# Predict on test set
X_test   = test_df[FEATURE_COLS].values
X_test_s = if_scaler.transform(X_test)

y_true   = np.where(test_df['label'] == 'normal', 0, 1)  # 1 = anomaly
y_pred   = np.where(if_model.predict(X_test_s) == -1, 1, 0)
raw_s    = if_model.decision_function(X_test_s)
risk_s   = np.clip((0.5 - raw_s) * 100, 0, 100) / 100.0

tp = int(np.sum((y_true==1) & (y_pred==1)))
fp = int(np.sum((y_true==0) & (y_pred==1)))
tn = int(np.sum((y_true==0) & (y_pred==0)))
fn = int(np.sum((y_true==1) & (y_pred==0)))
tpr = tp/(tp+fn) if (tp+fn)>0 else 0
fpr = fp/(fp+tn) if (fp+tn)>0 else 0
auc = roc_auc_score(y_true, risk_s)

print(f"\n{'Metric':<28} {'Value':<12} {'Target':<12} {'Status'}")
print("-" * 65)
print(f"{'True Positive Rate (TPR)':<28} {tpr:<12.4f} {'≥ 0.90':<12} {'PASS' if tpr>=0.90 else 'FAIL'}")
print(f"{'False Positive Rate (FPR)':<28} {fpr:<12.4f} {'≤ 0.05':<12} {'PASS' if fpr<=0.05 else 'FAIL'}")
print(f"{'AUC-ROC':<28} {auc:<12.4f} {'≥ 0.90':<12} {'PASS' if auc>=0.90 else 'FAIL'}")
print(f"{'F1 Score':<28} {f1_score(y_true,y_pred,zero_division=0):<12.4f}")
print(f"{'Precision':<28} {precision_score(y_true,y_pred,zero_division=0):<12.4f}")
print(f"{'Recall':<28} {recall_score(y_true,y_pred,zero_division=0):<12.4f}")
print(f"\nConfusion Matrix:")
print(f"  TP={tp}  FP={fp}")
print(f"  FN={fn}  TN={tn}")

# %% [markdown]
# ## Cell 8 — Isolation Forest visualisations

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1. Risk score distribution by label
risk_all = np.clip((0.5 - if_model.decision_function(if_scaler.transform(test_df[FEATURE_COLS].values))) * 100, 0, 100)
for label, color in [('normal','steelblue'), ('low_risk_anomaly','orange'), ('high_risk_anomaly','red')]:
    mask = test_df['label'] == label
    if mask.sum() > 0:
        axes[0].hist(risk_all[mask], bins=30, alpha=0.7, label=label, color=color)
axes[0].axvline(50, color='orange', linestyle='--', linewidth=1, label='Soft warning (50)')
axes[0].axvline(75, color='red',    linestyle='--', linewidth=1, label='Alert (75)')
axes[0].set_title('IF Risk Score Distribution')
axes[0].set_xlabel('Risk Score (0-100)')
axes[0].set_ylabel('Count')
axes[0].legend(fontsize=8)

# 2. ROC curve
fpr_c, tpr_c, _ = roc_curve(y_true, risk_s)
axes[1].plot(fpr_c, tpr_c, color='steelblue', lw=2, label=f'ROC (AUC = {auc:.3f})')
axes[1].plot([0,1],[0,1], 'k--', lw=1)
axes[1].set_title('ROC Curve — Isolation Forest')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

# 3. Confusion matrix heatmap
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Predicted Normal','Predicted Anomaly'],
            yticklabels=['True Normal','True Anomaly'])
axes[2].set_title('Confusion Matrix — IF')

plt.tight_layout()
plt.savefig('../models/if_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved to ../models/if_evaluation.png")

# %% [markdown]
# ## Cell 9 — Train Shallow Autoencoder (SECONDARY MODEL)

# %%
print("=" * 50)
print("  Training Shallow Autoencoder (Secondary Model)")
print("  Architecture: 19 → 10 → 19")
print("=" * 50)

normal_train_ae = train_df[train_df['label'] == 'normal']
print(f"\nTraining on {len(normal_train_ae):,} normal samples...")

# Scale features (separate scaler from IF)
ae_scaler  = StandardScaler()
X_ae_train = normal_train_ae[FEATURE_COLS].values
X_ae_scaled= ae_scaler.fit_transform(X_ae_train)

# Train autoencoder (19 → 10 → 19)
ae_model = MLPRegressor(
    hidden_layer_sizes=(10,),   # bottleneck layer
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False
)
ae_model.fit(X_ae_scaled, X_ae_scaled)   # X → X (reconstruct itself)

# Calculate error threshold from training data
X_ae_reconstructed = ae_model.predict(X_ae_scaled)
train_errors = np.mean((X_ae_scaled - X_ae_reconstructed) ** 2, axis=1)
ae_threshold = float(np.mean(train_errors) + 2 * np.std(train_errors))

print(f"\nTraining complete!")
print(f"Iterations used      : {ae_model.n_iter_}")
print(f"Mean training error  : {np.mean(train_errors):.6f}")
print(f"Std training error   : {np.std(train_errors):.6f}")
print(f"Anomaly threshold    : {ae_threshold:.6f}  (mean + 2*std)")

# Save model, scaler, and threshold
with open("../models/ae_model.pkl", "wb") as f:
    pickle.dump(ae_model, f)
with open("../models/ae_scaler.pkl", "wb") as f:
    pickle.dump(ae_scaler, f)
with open("../models/ae_threshold.pkl", "wb") as f:
    pickle.dump(ae_threshold, f)

print("\nAll AE files saved to ../models/")

# %% [markdown]
# ## Cell 10 — Evaluate Autoencoder

# %%
print("=" * 50)
print("  Autoencoder — Evaluation")
print("=" * 50)

X_test_ae  = test_df[FEATURE_COLS].values
X_test_aes = ae_scaler.transform(X_test_ae)
X_test_rec = ae_model.predict(X_test_aes)
ae_errors  = np.mean((X_test_aes - X_test_rec) ** 2, axis=1)

y_true_ae  = np.where(test_df['label'] == 'normal', 0, 1)
y_pred_ae  = np.where(ae_errors > ae_threshold, 1, 0)
risk_ae    = np.clip((ae_errors / ae_threshold) * 50, 0, 100) / 100.0

tp2 = int(np.sum((y_true_ae==1) & (y_pred_ae==1)))
fp2 = int(np.sum((y_true_ae==0) & (y_pred_ae==1)))
tn2 = int(np.sum((y_true_ae==0) & (y_pred_ae==0)))
fn2 = int(np.sum((y_true_ae==1) & (y_pred_ae==0)))
tpr2 = tp2/(tp2+fn2) if (tp2+fn2)>0 else 0
fpr2 = fp2/(fp2+tn2) if (fp2+tn2)>0 else 0
auc2 = roc_auc_score(y_true_ae, risk_ae)

print(f"\n{'Metric':<28} {'Value':<12} {'Target':<12} {'Status'}")
print("-" * 65)
print(f"{'True Positive Rate (TPR)':<28} {tpr2:<12.4f} {'≥ 0.90':<12} {'PASS' if tpr2>=0.90 else 'FAIL'}")
print(f"{'False Positive Rate (FPR)':<28} {fpr2:<12.4f} {'≤ 0.05':<12} {'PASS' if fpr2<=0.05 else 'FAIL'}")
print(f"{'AUC-ROC':<28} {auc2:<12.4f} {'≥ 0.90':<12} {'PASS' if auc2>=0.90 else 'FAIL'}")
print(f"{'F1 Score':<28} {f1_score(y_true_ae,y_pred_ae,zero_division=0):<12.4f}")
print(f"{'Precision':<28} {precision_score(y_true_ae,y_pred_ae,zero_division=0):<12.4f}")
print(f"{'Recall':<28} {recall_score(y_true_ae,y_pred_ae,zero_division=0):<12.4f}")
print(f"\nConfusion Matrix:")
print(f"  TP={tp2}  FP={fp2}")
print(f"  FN={fn2}  TN={tn2}")

# %% [markdown]
# ## Cell 11 — Autoencoder visualisations

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1. Reconstruction error distribution
for label, color in [('normal','steelblue'), ('low_risk_anomaly','orange'), ('high_risk_anomaly','red')]:
    mask = test_df['label'] == label
    if mask.sum() > 0:
        axes[0].hist(ae_errors[mask], bins=40, alpha=0.7, label=label, color=color)
axes[0].axvline(ae_threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({ae_threshold:.4f})')
axes[0].set_title('AE Reconstruction Error')
axes[0].set_xlabel('MSE Reconstruction Error')
axes[0].set_ylabel('Count')
axes[0].legend(fontsize=8)

# 2. ROC curve
fpr_c2, tpr_c2, _ = roc_curve(y_true_ae, risk_ae)
axes[1].plot(fpr_c2, tpr_c2, color='darkorange', lw=2, label=f'ROC (AUC = {auc2:.3f})')
axes[1].plot([0,1],[0,1], 'k--', lw=1)
axes[1].set_title('ROC Curve — Autoencoder')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

# 3. Confusion matrix
cm2 = confusion_matrix(y_true_ae, y_pred_ae)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
            xticklabels=['Predicted Normal','Predicted Anomaly'],
            yticklabels=['True Normal','True Anomaly'])
axes[2].set_title('Confusion Matrix — AE')

plt.tight_layout()
plt.savefig('../models/ae_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 12 — Combined composite score test

# %%
print("=" * 50)
print("  Composite Score (IF 60% + AE 40%)")
print("=" * 50)

X_test_all = test_df[FEATURE_COLS].values

# IF risk scores
if_raw   = if_model.decision_function(if_scaler.transform(X_test_all))
if_risk  = np.clip((0.5 - if_raw) * 100, 0, 100)

# AE risk scores
ae_err   = np.mean((ae_scaler.transform(X_test_all) -
                    ae_model.predict(ae_scaler.transform(X_test_all)))**2, axis=1)
ae_risk  = np.clip((ae_err / ae_threshold) * 50, 0, 100)

# Composite
composite = (if_risk * 0.60) + (ae_risk * 0.40)
y_true_c  = np.where(test_df['label'] == 'normal', 0, 1)
y_pred_c  = np.where(composite >= 75, 1, 0)
auc_c     = roc_auc_score(y_true_c, composite / 100.0)

print(f"\nComposite AUC-ROC : {auc_c:.4f}")
print(f"F1 Score          : {f1_score(y_true_c, y_pred_c, zero_division=0):.4f}")
print(f"\nSample predictions (first 10 test rows):")
print(f"{'Label':<22} {'IF Risk':<12} {'AE Risk':<12} {'Composite':<12} {'Decision'}")
print("-" * 70)
for i in range(min(10, len(test_df))):
    row_label = test_df['label'].iloc[i]
    decision  = 'ALERT' if composite[i] >= 75 else ('WARNING' if composite[i] >= 50 else 'normal')
    print(f"{row_label:<22} {if_risk[i]:<12.1f} {ae_risk[i]:<12.1f} {composite[i]:<12.1f} {decision}")

# %% [markdown]
# ## Cell 13 — Final summary and model file check

# %%
print("=" * 55)
print("  FINAL SUMMARY — All Models Trained & Saved")
print("=" * 55)

model_files = [
    '../models/if_model.pkl',
    '../models/if_scaler.pkl',
    '../models/ae_model.pkl',
    '../models/ae_scaler.pkl',
    '../models/ae_threshold.pkl',
]

print("\nSaved model files:")
for path in model_files:
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    print(f"  {'OK' if exists else 'MISSING':<8} {path:<40} {size/1024:.1f} KB")

print(f"\nIsolation Forest  → AUC={auc:.4f}  TPR={tpr:.4f}  FPR={fpr:.4f}")
print(f"Autoencoder       → AUC={auc2:.4f}  TPR={tpr2:.4f}  FPR={fpr2:.4f}")
print(f"Composite Score   → AUC={auc_c:.4f}")
print(f"\nAll files ready for C3_activity_monitoring/models/")
print("Copy these .pkl files into your project's models/ folder.")
