# C4 — Productivity Prediction

Predicts employee productivity scores using a trained ML model based on
activity features collected by C3.

## Owner
Team Member 4

## Dependencies
```
scikit-learn, tensorflow or torch, shap, lime, pandas, numpy, pymongo
```

## Interface
- `start_productivity_logger(user_id, db_client, shutdown_event)` — starts loop

## Prediction Pipeline
```
FeatureVector (from C3)
    ↓
Productivity model inference
    ↓
ProductivityDocument → MongoDB::productivity_scores
    ↓
SHAP / LIME explanation
    ↓
WebSocket → dashboard
```
