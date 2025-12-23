import pandas as pd
import re
from collections import defaultdict

def classify_table_type(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "unknown"

    text = " ".join(df.astype(str).fillna("").values.ravel()).lower()

    scores = defaultdict(int)

    LAB_KEYWORDS = [
        "hemoglobin", "hb", "wbc", "rbc", "platelet",
        "bilirubin", "creatinine", "urea", "sodium", "potassium",
        "ast", "alt", "alp", "hba1c", "tsh", "cholesterol", "ldl", "hdl"
    ]

    ECG_KEYWORDS = [
        "pr interval", "pr(", "qrs", "qt", "qtc",
        "st segment", "axis", "bpm", "heart rate", "sinus"
    ]

    USG_KEYWORDS = [
        "bpd", "hc", "ac", "fl", "efw",
        "gestational age", "crl", "fetal"
    ]

    CT_MRI_KEYWORDS = [
        "lesion", "slice", "series", "contrast",
        "t1", "t2", "flair", "roi", "hu", "attenuation"
    ]

    def score_keywords(words, label):
        for w in words:
            if w in text:
                scores[label] += 1

    score_keywords(LAB_KEYWORDS, "lab")
    score_keywords(ECG_KEYWORDS, "ecg")
    score_keywords(USG_KEYWORDS, "ultrasound")
    score_keywords(CT_MRI_KEYWORDS, "ct_mri")

    if not scores:
        return "unknown"

    # pick highest score
    best_type, best_score = max(scores.items(), key=lambda x: x[1])

    # weak signal --> unknown
    if best_score < 2:
        return "unknown"

    return best_type