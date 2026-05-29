import pandas as pd
import numpy as np

def add_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["NotificationCTR"] = (
        df["NoOfNotificationsClicked"] / df["NoOfNotificationsViewed"].replace(0, np.nan)
    ).fillna(0)

    df["EngagementTrend_30_vs_60"] = (
        (df["AppLaunches_0_30"] + 1) / (df["AppLaunches_30_60"] + 1)
    )

    consultation_cols = ["HasTakenChat", "HasTakenCall", "HasTakenOther"]
    df["ConsultationCount"] = df[consultation_cols].sum(axis=1)

    df["LowWalletFlag"] = (df["CurrentWalletBalance"] < 100).astype(int)
    df["DecliningEngagementFlag"] = (df["EngagementTrend_30_vs_60"] < 0.65).astype(int)
    df["LowNotificationCTRFlag"] = (df["NotificationCTR"] < 0.08).astype(int)

    return df
