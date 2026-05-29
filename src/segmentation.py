import pandas as pd

def assign_churn_segment(row: pd.Series) -> str:
    high_usage = row.get("AppLaunches_0_30", 0) >= 8
    low_wallet = row.get("CurrentWalletBalance", 0) < 100
    no_consult = row.get("ConsultationCount", 0) == 0
    high_wallet = row.get("CurrentWalletBalance", 0) >= 500

    if high_usage and low_wallet and no_consult:
        return "Price Sensitive"

    if row.get("HasUninstalled", 0) == 1 or row.get("HasPushUnregistered", 0) == 1 or row.get("HadNegativeInteraction", 0) == 1:
        return "Feature/Tech Frustrated"

    if high_wallet and high_usage and no_consult:
        return "Value Skeptic"

    if row.get("DecliningEngagementFlag", 0) == 1 or row.get("LowNotificationCTRFlag", 0) == 1:
        return "Drifting"

    return "General Risk"


def recommended_action(segment: str) -> str:
    actions = {
        "Price Sensitive": "Send discount or promotional wallet offer",
        "Feature/Tech Frustrated": "Send support or bug-fix communication",
        "Value Skeptic": "Send free consultation offer or value proof campaign",
        "Drifting": "Send content-based re-engagement campaign",
        "General Risk": "Send generic retention reminder",
    }
    return actions.get(segment, "Send generic retention reminder")
