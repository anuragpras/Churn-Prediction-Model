import os
import yaml
import joblib
import pandas as pd

from feature_engineering import add_churn_features
from segmentation import assign_churn_segment, recommended_action


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def score_bucket(score):
    if score >= 0.80:
        return "0.80 - 1.00"
    if score >= 0.60:
        return "0.60 - 0.80"
    if score >= 0.40:
        return "0.40 - 0.60"
    if score >= 0.20:
        return "0.20 - 0.40"
    return "0.00 - 0.20"


def main():
    cfg = load_config()
    model_path = os.path.join(cfg["paths"]["model_dir"], "churn_model.pkl")
    output_dir = cfg["paths"]["output_dir"]

    model = joblib.load(model_path)
    df = add_churn_features(pd.read_csv(cfg["paths"]["predict_data"]))

    id_cols = ["UserID", "SignupDate"]
    features = [c for c in df.columns if c not in id_cols + ["Churn_30D", "Churn_60D", "Churn_90D"]]

    scores = model.predict_proba(df[features])[:, 1]

    out = df[id_cols].copy()
    out["Churn_Score"] = scores.round(4)
    out["Score_Bucket"] = out["Churn_Score"].apply(score_bucket)
    df["Churn_Score"] = out["Churn_Score"]
    df["Score_Bucket"] = out["Score_Bucket"]
    out["Churn_Segment"] = df.apply(assign_churn_segment, axis=1)
    out["Recommended_Action"] = out["Churn_Segment"].apply(recommended_action)

    out = out.sort_values("Churn_Score", ascending=False)
    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(os.path.join(output_dir, "new_batch_churn_scores.csv"), index=False)

    print("Prediction complete.")


if __name__ == "__main__":
    main()
