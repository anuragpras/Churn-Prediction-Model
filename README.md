# Churn Prediction Model

## Overview

This repository contains an open-source machine learning pipeline for predicting whether a user is likely to **uninstall the app or become inactive** within a selected future window.
<img width="1536" height="1024" alt="churn" src="https://github.com/user-attachments/assets/74810129-9bce-4a1c-a6f0-023a297af5c4" />


Supported churn windows:

- 30 days
- 60 days
- 90 days

The model is designed for batch scoring and retention campaign planning. It generates user-level churn scores, churn-risk segments, feature importance, decile analysis, score bucket summaries, and campaign-ready output files.

---

## Objective

Build a binary classification model to predict whether a user will churn in the next 30, 60, or 90 days.

```text
1 = User uninstalled the app or became inactive
0 = User remained active
```

---

## Business Logic

User churn is mainly driven by:

1. **Declining engagement**  
   Fewer recent app launches compared to previous periods.

2. **Low value realization**  
   Users have not taken consultations or completed meaningful product actions.

3. **Financial friction**  
   Low wallet balance, no recent recharge, or price-sensitive behavior.

---

## Feature Engineering

Raw data alone does not capture behavior change. This project creates derived churn signals.

### Engagement Trend

```text
EngagementTrend_30_vs_60 = (AppLaunches_0_30 + 1) / (AppLaunches_30_60 + 1)
```

A value below 1 indicates declining engagement.

### Notification Engagement

```text
NotificationCTR = NoOfNotificationsClicked / NoOfNotificationsViewed
```

Low CTR can indicate notification fatigue or disengagement.

### Recency Metrics

The model uses:

- DaysSinceLastConsultation
- DaysSinceLastRecharge

Increasing gaps usually indicate reduced activity.

---

## User Segmentation

Users are grouped by primary churn reason, not only churn score.

| Segment | Logic | Recommended Action |
|---|---|---|
| Price Sensitive | High usage, low wallet balance, no consultations | Send discount or promotional wallet offer |
| Feature/Tech Frustrated | Uninstall, push unregister, or negative interaction signals | Send support or bug-fix communication |
| Value Skeptic | High wallet balance and usage but no consultations | Send free consultation or value proof campaign |
| Drifting | Declining engagement or low notification CTR | Send content-based re-engagement campaign |
| General Risk | Moderate risk without a dominant reason | Send generic retention reminder |

---

## Project Structure

```text
churn-prediction-model/
│
├── data/
│   ├── sample_train.csv
│   └── sample_predict.csv
│
├── models/
│   └── MODEL_CARD.md
│
├── notebooks/
│   └── README.md
│
├── outputs/
│   ├── all_scored_users.csv
│   ├── top_10pct_churn_risk_users.csv
│   └── churn_model_analysis_report.xlsx
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── feature_engineering.py
│   ├── segmentation.py
│   └── generate_sample_data.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Input Data

### Training Data

The training file should contain user-level features and one or more target columns:

```text
Churn_30D
Churn_60D
Churn_90D
```

Select the target in `config.yaml`:

```yaml
project:
  target_window: Churn_30D
```

### Prediction Data

The prediction file should contain the same feature columns but does not need target columns.

Required ID columns:

```text
UserID
SignupDate
```

---

## Installation

```bash
git clone https://github.com/yourusername/churn-prediction-model.git
cd churn-prediction-model
pip install -r requirements.txt
```

---

## How To Run

### Step 1: Use sample data or replace it

```text
data/sample_train.csv
data/sample_predict.csv
```

### Step 2: Select churn window

Edit `config.yaml`:

```yaml
target_window: Churn_30D
```

Available options:

```text
Churn_30D
Churn_60D
Churn_90D
```

### Step 3: Train and score

```bash
python src/train.py
```

Generated outputs:

```text
models/churn_model.pkl
outputs/all_scored_users.csv
outputs/top_10pct_churn_risk_users.csv
outputs/churn_model_analysis_report.xlsx
```

### Step 4: Score a new batch

After training:

```bash
python src/predict.py
```

Generated output:

```text
outputs/new_batch_churn_scores.csv
```

---

## Output Files

### all_scored_users.csv

Contains:

- UserID
- SignupDate
- Churn_Score
- Score_Bucket
- Churn_Segment
- Recommended_Action

### top_10pct_churn_risk_users.csv

Highest-risk users for retention campaigns.

### churn_model_analysis_report.xlsx

Contains:

- Model_Metrics
- Threshold_Analysis
- Feature_Importance
- Train_Deciles
- Test_Deciles
- Predict_Deciles
- Score_Bucket_Summary
- Feature_Bucket_Analysis

---

## Production Rollout

### Phase 1: Model Validation

Train on historical data and test on the most recent period.

Recommended setup:

```text
Train: 90 to 30 days ago
Test: Most recent 30 days
```

Primary metric:

```text
Recall
```

Recall is important because the goal is to capture as many churn-risk users as possible.

### Phase 2: Shadow Deployment

Run the model without triggering user actions.

Track:

- Predicted churn score
- Actual churn outcome
- Segment accuracy
- Campaign eligibility
- False positives and false negatives

Recommended shadow period:

```text
2 weeks
```

### Phase 3: Integration and Automation

Suggested pipeline:

```text
SQL ETL
   ↓
Python Model Inference
   ↓
User Table / CRM Table
   ↓
Automated Retention Campaigns
```

Example rule:

```text
If Churn_Score > 0.80 and Churn_Segment = Price Sensitive,
send a 20% discount offer.
```

---

## Future Improvements

- SHAP explainability
- Hyperparameter tuning
- Multi-window churn modeling
- Survival analysis
- Uplift modeling
- MLflow experiment tracking
- Model monitoring
- Automated retraining
- Dashboard integration

---

## Disclaimer

The sample data is synthetic and does not contain real user information.

---

## License

MIT License
