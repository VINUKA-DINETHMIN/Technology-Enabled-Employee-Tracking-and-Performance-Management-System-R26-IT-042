#!/usr/bin/env python
"""Comprehensive test of model loading and feature building"""
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Load model
obj = joblib.load('C4_productivity_prediction/src/employee_productivity_forecast_model.joblib')
model = obj.get('model')
feat_cols = obj.get('feature_columns')

print(f"[OK] Model loaded: {type(model).__name__}")
print(f"[OK] Features expected: {len(feat_cols)}")

# Test 1: Predict with all baseline features
print("\n--- Test 1: Baseline Features ---")
test_data_dict = {}
for feat in feat_cols:
    if feat in ['employee_id', 'department', 'role']:
        test_data_dict[feat] = "Test"
    else:
        test_data_dict[feat] = 50.0

test_df = pd.DataFrame([test_data_dict])
test_df = test_df[feat_cols]  # Ensure correct order

try:
    pred = model.predict(test_df)
    print(f"[OK] Prediction (baseline): {pred[0]:.2f}")
except Exception as e:
    print(f"[ERROR] Prediction failed: {e}")
    sys.exit(1)

# Test 2: High productivity scenario
print("\n--- Test 2: High Productivity ---")
test_data_dict = {}
for feat in feat_cols:
    if feat in ['employee_id', 'department', 'role']:
        test_data_dict[feat] = "Test"
    elif 'productivity' in feat or 'workload' in feat:
        test_data_dict[feat] = 80.0
    elif 'completion_rate' in feat:
        test_data_dict[feat] = 0.9
    elif 'tasks' in feat:
        test_data_dict[feat] = 10.0
    else:
        test_data_dict[feat] = 50.0

test_df = pd.DataFrame([test_data_dict])
test_df = test_df[feat_cols]

try:
    pred = model.predict(test_df)
    print(f"[OK] Prediction (high): {pred[0]:.2f}")
except Exception as e:
    print(f"[ERROR] Prediction failed: {e}")
    sys.exit(1)

# Test 3: Low productivity scenario
print("\n--- Test 3: Low Productivity ---")
test_data_dict = {}
for feat in feat_cols:
    if feat in ['employee_id', 'department', 'role']:
        test_data_dict[feat] = "Test"
    elif 'productivity' in feat or 'workload' in feat:
        test_data_dict[feat] = 20.0
    elif 'completion_rate' in feat:
        test_data_dict[feat] = 0.2
    elif 'tasks' in feat:
        test_data_dict[feat] = 2.0
    else:
        test_data_dict[feat] = 50.0

test_df = pd.DataFrame([test_data_dict])
test_df = test_df[feat_cols]

try:
    pred = model.predict(test_df)
    print(f"[OK] Prediction (low): {pred[0]:.2f}")
except Exception as e:
    print(f"[ERROR] Prediction failed: {e}")
    sys.exit(1)

print("\n[SUCCESS] All tests passed!")



