"""
test_anomaly_detection.py
Test suite for Phase 1 Anomaly Detection module and edge case validation.
"""

import pandas as pd
from modules.anomaly_detection import (
    detect_transaction_anomalies,
    detect_category_anomalies,
    detect_trend_anomalies,
    get_anomaly_fallback_explanation,
)


def run_tests():
    print("--- Running Anomaly Detection Test Suite ---")

    # 1. Edge Case Test: Empty DataFrame
    empty_df = pd.DataFrame(columns=["Date", "Type", "Category", "Description", "Amount"])
    tx_empty = detect_transaction_anomalies(empty_df)
    cat_empty = detect_category_anomalies(empty_df)
    trend_empty = detect_trend_anomalies(empty_df)
    assert len(tx_empty) == 0, "Empty DF should yield 0 transaction anomalies"
    assert len(cat_empty) == 0, "Empty DF should yield 0 category anomalies"
    assert len(trend_empty) == 0, "Empty DF should yield 0 trend anomalies"
    print("[PASS] Passed: Empty DataFrame test")

    # 2. Edge Case Test: Insufficient category data & identical amounts
    test_data = [
        {"Date": "2026-06-01", "Type": "Expense", "Category": "Rent", "Description": "June Rent", "Amount": 1000.0},
        {"Date": "2026-06-02", "Type": "Expense", "Category": "Rent", "Description": "July Rent", "Amount": 1000.0}, # Identical std=0
        {"Date": "2026-06-03", "Type": "Expense", "Category": "SoloCat", "Description": "Solo item", "Amount": 5000.0}, # Insufficient data < 3
        # Category "Supplies" with 10 normal items (~$100) and 1 extreme outlier ($5000)
        {"Date": "2026-06-04", "Type": "Expense", "Category": "Supplies", "Description": "Paper", "Amount": 100.0},
        {"Date": "2026-06-05", "Type": "Expense", "Category": "Supplies", "Description": "Pens", "Amount": 105.0},
        {"Date": "2026-06-06", "Type": "Expense", "Category": "Supplies", "Description": "Folder", "Amount": 95.0},
        {"Date": "2026-06-07", "Type": "Expense", "Category": "Supplies", "Description": "Ink", "Amount": 100.0},
        {"Date": "2026-06-08", "Type": "Expense", "Category": "Supplies", "Description": "Desk", "Amount": 98.0},
        {"Date": "2026-06-09", "Type": "Expense", "Category": "Supplies", "Description": "Stapler", "Amount": 102.0},
        {"Date": "2026-06-10", "Type": "Expense", "Category": "Supplies", "Description": "Clips", "Amount": 97.0},
        {"Date": "2026-06-11", "Type": "Expense", "Category": "Supplies", "Description": "Tape", "Amount": 101.0},
        {"Date": "2026-06-12", "Type": "Expense", "Category": "Supplies", "Description": "Markers", "Amount": 99.0},
        {"Date": "2026-06-13", "Type": "Expense", "Category": "Supplies", "Description": "Postits", "Amount": 103.0},
        {"Date": "2026-06-14", "Type": "Expense", "Category": "Supplies", "Description": "Extreme Outlier Device", "Amount": 5000.0},
    ]

    df_test = pd.DataFrame(test_data)
    df_test["Date"] = pd.to_datetime(df_test["Date"])

    anomalies = detect_transaction_anomalies(df_test, z_medium=2.5, z_high=3.0, min_samples=3)

    assert len(anomalies) == 1, f"Expected exactly 1 anomaly, got {len(anomalies)}"
    flagged = anomalies[0]
    assert flagged["Category"] == "Supplies", "Flagged category should be Supplies"
    assert flagged["Amount"] == 5000.0, "Flagged amount should be 5000.0"
    assert flagged["Severity"] == "High", "Outlier Z-score should be classified as High severity"
    print("[PASS] Passed: Outlier detection & insufficient data filtering test")

    # 3. Rule-Based Fallback Explanation Test
    explanation = get_anomaly_fallback_explanation(anomalies, [], [])
    assert "High Priority Transaction Review" in explanation
    assert "Extreme Outlier Device" in explanation
    assert "fraud" not in explanation.lower(), "Must never use the word fraud"
    print("[PASS] Passed: Fallback explanation formatting test")

    # 4. Multi-month Category & Trend Anomaly Test
    multi_month = [
        {"Date": "2026-06-01", "Type": "Income", "Category": "Sales", "Description": "Rev June", "Amount": 10000.0},
        {"Date": "2026-06-02", "Type": "Expense", "Category": "Marketing", "Description": "Ads June", "Amount": 500.0},
        {"Date": "2026-07-01", "Type": "Income", "Category": "Sales", "Description": "Rev July", "Amount": 4000.0}, # 60% revenue drop
        {"Date": "2026-07-02", "Type": "Expense", "Category": "Marketing", "Description": "Ads July", "Amount": 2500.0}, # 400% expense surge
    ]
    df_multi = pd.DataFrame(multi_month)
    df_multi["Date"] = pd.to_datetime(df_multi["Date"])

    cat_anom = detect_category_anomalies(df_multi, threshold_pct=30.0)
    trend_anom = detect_trend_anomalies(df_multi, threshold_pct=25.0)

    assert len(cat_anom) > 0, "Should detect Marketing spending spike"
    assert len(trend_anom) >= 2, "Should detect Revenue Drop and Expense Spike"
    print("[PASS] Passed: Category spike and Trend anomaly detection test")

    print("\n=== ALL ANOMALY DETECTION TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_tests()
