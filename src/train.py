import os
import yaml
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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


def decile_summary(scores, actual=None):
    df = pd.DataFrame({"Score": scores})
    df["Decile"] = 10 - pd.qcut(df["Score"].rank(method="first"), 10, labels=False)

    out = df.groupby("Decile").agg(
        Users=("Score", "count"),
        Min_Score=("Score", "min"),
        Max_Score=("Score", "max"),
        Avg_Score=("Score", "mean")
    ).reset_index()

    if actual is not None:
        df["Actual"] = actual.values
        churn = df.groupby("Decile")["Actual"].agg(["sum", "mean"]).reset_index()
        churn.columns = ["Decile", "Churned_Users", "Churn_Rate"]
        out = out.merge(churn, on="Decile", how="left")
        out["Capture_Rate"] = out["Churned_Users"] / out["Churned_Users"].sum()

    return out.round(4)


def main():
    cfg = load_config()
    target = cfg["project"]["target_window"]
    train_path = cfg["paths"]["train_data"]
    predict_path = cfg["paths"]["predict_data"]
    output_dir = cfg["paths"]["output_dir"]
    model_dir = cfg["paths"]["model_dir"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train_df = add_churn_features(pd.read_csv(train_path))
    predict_df = add_churn_features(pd.read_csv(predict_path))

    id_cols = ["UserID", "SignupDate"]
    target_cols = ["Churn_30D", "Churn_60D", "Churn_90D"]
    features = [c for c in train_df.columns if c not in id_cols + target_cols]

    X = train_df[features]
    y = train_df[target]
    X_pred = predict_df[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
        stratify=y
    )

    numeric_features = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ]
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=cfg["model"]["random_state"],
        n_jobs=-1,
        verbose=-1
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    test_scores = pipeline.predict_proba(X_test)[:, 1]
    train_scores = pipeline.predict_proba(X_train)[:, 1]
    pred_scores = pipeline.predict_proba(X_pred)[:, 1]

    threshold = cfg["model"]["threshold"]
    test_preds = (test_scores >= threshold).astype(int)

    metrics = pd.DataFrame([{
        "Target_Window": target,
        "AUC_ROC": roc_auc_score(y_test, test_scores),
        "Accuracy": accuracy_score(y_test, test_preds),
        "Precision": precision_score(y_test, test_preds),
        "Recall": recall_score(y_test, test_preds),
        "F1_Score": f1_score(y_test, test_preds),
        "Threshold": threshold
    }]).round(4)

    pred_out = predict_df[id_cols].copy()
    pred_out["Churn_Score"] = pred_scores.round(4)
    pred_out["Score_Bucket"] = pred_out["Churn_Score"].apply(score_bucket)

    predict_df["Churn_Score"] = pred_out["Churn_Score"]
    predict_df["Score_Bucket"] = pred_out["Score_Bucket"]
    pred_out["Churn_Segment"] = predict_df.apply(assign_churn_segment, axis=1)
    pred_out["Recommended_Action"] = pred_out["Churn_Segment"].apply(recommended_action)
    pred_out = pred_out.sort_values("Churn_Score", ascending=False)

    top_percent = cfg["outputs"]["top_percent"]
    top_n = max(1, int(len(pred_out) * top_percent / 100))

    pred_out.to_csv(os.path.join(output_dir, "all_scored_users.csv"), index=False)
    pred_out.head(top_n).to_csv(os.path.join(output_dir, f"top_{top_percent}pct_churn_risk_users.csv"), index=False)

    feature_importance = pd.DataFrame({
        "Feature": pipeline.named_steps["preprocess"].get_feature_names_out(),
        "Importance": pipeline.named_steps["model"].feature_importances_
    })
    feature_importance["Importance_Percent"] = feature_importance["Importance"] / feature_importance["Importance"].sum() * 100
    feature_importance = feature_importance.sort_values("Importance_Percent", ascending=False).head(30)

    predict_df["Score_Bucket"] = pred_out.set_index("UserID").loc[predict_df["UserID"], "Score_Bucket"].values
    feature_bucket = predict_df.groupby("Score_Bucket").agg(
        Users=("UserID", "count"),
        Avg_CurrentWalletBalance=("CurrentWalletBalance", "mean"),
        Avg_AppLaunches_0_30=("AppLaunches_0_30", "mean"),
        Avg_EngagementTrend=("EngagementTrend_30_vs_60", "mean"),
        Avg_NotificationCTR=("NotificationCTR", "mean"),
        Avg_DaysSinceLastConsultation=("DaysSinceLastConsultation", "mean"),
        Avg_DaysSinceLastRecharge=("DaysSinceLastRecharge", "mean"),
        Avg_ConsultationCount=("ConsultationCount", "mean")
    ).reset_index().round(4)

    with pd.ExcelWriter(os.path.join(output_dir, "churn_model_analysis_report.xlsx"), engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Model_Metrics", index=False)
        feature_importance.to_excel(writer, sheet_name="Feature_Importance", index=False)
        decile_summary(train_scores, y_train).to_excel(writer, sheet_name="Train_Deciles", index=False)
        decile_summary(test_scores, y_test).to_excel(writer, sheet_name="Test_Deciles", index=False)
        decile_summary(pred_scores).to_excel(writer, sheet_name="Predict_Deciles", index=False)
        pred_out.groupby(["Score_Bucket", "Churn_Segment"]).size().reset_index(name="Users").to_excel(writer, sheet_name="Score_Bucket_Summary", index=False)
        feature_bucket.to_excel(writer, sheet_name="Feature_Bucket_Analysis", index=False)

    joblib.dump(pipeline, os.path.join(model_dir, "churn_model.pkl"))

    print("Model training and scoring complete.")


if __name__ == "__main__":
    main()
