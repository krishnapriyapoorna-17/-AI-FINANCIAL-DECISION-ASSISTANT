"""
test_risk_forecast.py
Unit test suite for Phase 2 Risk & Forecasting module.
"""

import pandas as pd
from modules.risk_forecast import (
    generate_short_term_forecast,
    detect_financial_risks,
    get_risk_forecast_fallback_explanation,
)


def run_tests():
    print("--- Running Risk & Forecast Test Suite ---")

    # 1. Test Empty Dataset
    empty_df = pd.DataFrame(columns=["Date", "Type", "Category", "Description", "Amount"])
    fc_empty = generate_short_term_forecast(empty_df)
    risk_empty = detect_financial_risks(empty_df, fc_empty)

    assert fc_empty["has_sufficient_data"] is False, "Empty DF should return has_sufficient_data=False"
    assert "Not enough historical data" in fc_empty["message"]
    assert risk_empty["has_data"] is False
    print("[PASS] Passed: Empty dataset test")

    # 2. Test Insufficient Data (Single Month)
    single_month_data = [
        {"Date": "2026-06-01", "Type": "Income", "Category": "Sales", "Description": "Retainer", "Amount": 5000.0},
        {"Date": "2026-06-02", "Type": "Expense", "Category": "Rent", "Description": "Office Rent", "Amount": 1200.0},
    ]
    df_single = pd.DataFrame(single_month_data)
    df_single["Date"] = pd.to_datetime(df_single["Date"])
    fc_single = generate_short_term_forecast(df_single)

    assert fc_single["has_sufficient_data"] is False, "1 month of data should return has_sufficient_data=False"
    assert "Not enough historical data" in fc_single["message"]
    print("[PASS] Passed: Insufficient data (1 month) test")

    # 3. Test Normal Increasing Revenue & Forecast Generation
    multi_data = [
        {"Date": "2026-05-01", "Type": "Income", "Category": "Sales", "Description": "May Sales", "Amount": 8000.0},
        {"Date": "2026-05-02", "Type": "Expense", "Category": "Rent", "Description": "May Rent", "Amount": 2000.0},
        {"Date": "2026-06-01", "Type": "Income", "Category": "Sales", "Description": "June Sales", "Amount": 10000.0},
        {"Date": "2026-06-02", "Type": "Expense", "Category": "Rent", "Description": "June Rent", "Amount": 2000.0},
        {"Date": "2026-07-01", "Type": "Income", "Category": "Sales", "Description": "July Sales", "Amount": 12000.0},
        {"Date": "2026-07-02", "Type": "Expense", "Category": "Rent", "Description": "July Rent", "Amount": 2000.0},
    ]
    df_inc = pd.DataFrame(multi_data)
    df_inc["Date"] = pd.to_datetime(df_inc["Date"])
    fc_inc = generate_short_term_forecast(df_inc)
    risk_inc = detect_financial_risks(df_inc, fc_inc)

    assert fc_inc["has_sufficient_data"] is True, "3 months of data should generate forecast"
    assert len(fc_inc["forecast"]) == 2, "Forecast should project next 2 months"
    assert fc_inc["next_month"]["projected_revenue"] > 12000.0, "Projected revenue should follow upward trend"
    assert risk_inc["revenue_risk"]["severity"] == "Low"
    print("[PASS] Passed: Increasing revenue forecast & Low revenue risk test")

    # 4. Test Declining Revenue Risk
    declining_data = [
        {"Date": "2026-05-01", "Type": "Income", "Category": "Sales", "Description": "May Sales", "Amount": 15000.0},
        {"Date": "2026-05-02", "Type": "Expense", "Category": "Rent", "Description": "May Rent", "Amount": 2000.0},
        {"Date": "2026-06-01", "Type": "Income", "Category": "Sales", "Description": "June Sales", "Amount": 10000.0},
        {"Date": "2026-06-02", "Type": "Expense", "Category": "Rent", "Description": "June Rent", "Amount": 2000.0},
        {"Date": "2026-07-01", "Type": "Income", "Category": "Sales", "Description": "July Sales", "Amount": 5000.0},
        {"Date": "2026-07-02", "Type": "Expense", "Category": "Rent", "Description": "July Rent", "Amount": 2000.0},
    ]
    df_dec = pd.DataFrame(declining_data)
    df_dec["Date"] = pd.to_datetime(df_dec["Date"])
    fc_dec = generate_short_term_forecast(df_dec)
    risk_dec = detect_financial_risks(df_dec, fc_dec)

    assert risk_dec["revenue_risk"]["severity"] == "High", "Significant revenue drop should trigger High revenue risk"
    print("[PASS] Passed: Declining revenue risk test")

    # 5. Test Increasing Expenses Faster Than Revenue
    surge_exp_data = [
        {"Date": "2026-05-01", "Type": "Income", "Category": "Sales", "Description": "May Sales", "Amount": 10000.0},
        {"Date": "2026-05-02", "Type": "Expense", "Category": "Rent", "Description": "May Rent", "Amount": 2000.0},
        {"Date": "2026-06-01", "Type": "Income", "Category": "Sales", "Description": "June Sales", "Amount": 10200.0},
        {"Date": "2026-06-02", "Type": "Expense", "Category": "Rent", "Description": "June Rent", "Amount": 5000.0},
        {"Date": "2026-07-01", "Type": "Income", "Category": "Sales", "Description": "July Sales", "Amount": 10400.0},
        {"Date": "2026-07-02", "Type": "Expense", "Category": "Rent", "Description": "July Rent", "Amount": 9500.0},
    ]
    df_surge = pd.DataFrame(surge_exp_data)
    df_surge["Date"] = pd.to_datetime(df_surge["Date"])
    fc_surge = generate_short_term_forecast(df_surge)
    risk_surge = detect_financial_risks(df_surge, fc_surge)

    assert risk_surge["expense_risk"]["severity"] == "High", "Steep expense surge should trigger High expense risk"
    print("[PASS] Passed: Expense surge risk test")

    # 6. Test Projected Net Loss & Negative Cash Flow Risk
    loss_data = [
        {"Date": "2026-06-01", "Type": "Income", "Category": "Sales", "Description": "June Sales", "Amount": 5000.0},
        {"Date": "2026-06-02", "Type": "Expense", "Category": "Rent", "Description": "June Rent", "Amount": 7000.0},
        {"Date": "2026-07-01", "Type": "Income", "Category": "Sales", "Description": "July Sales", "Amount": 4000.0},
        {"Date": "2026-07-02", "Type": "Expense", "Category": "Rent", "Description": "July Rent", "Amount": 8000.0},
    ]
    df_loss = pd.DataFrame(loss_data)
    df_loss["Date"] = pd.to_datetime(df_loss["Date"])
    fc_loss = generate_short_term_forecast(df_loss)
    risk_loss = detect_financial_risks(df_loss, fc_loss)

    assert risk_loss["profitability_risk"]["severity"] == "High", "Net loss should trigger High profitability risk"
    assert risk_loss["cash_flow_risk"]["severity"] == "High", "Negative cash flow should trigger High cash flow risk"
    print("[PASS] Passed: Projected loss & cash flow risk test")

    # 7. Test Flat Financial Data
    flat_data = [
        {"Date": "2026-05-01", "Type": "Income", "Category": "Sales", "Description": "Sales", "Amount": 10000.0},
        {"Date": "2026-05-02", "Type": "Expense", "Category": "Rent", "Description": "Rent", "Amount": 5000.0},
        {"Date": "2026-06-01", "Type": "Income", "Category": "Sales", "Description": "Sales", "Amount": 10000.0},
        {"Date": "2026-06-02", "Type": "Expense", "Category": "Rent", "Description": "Rent", "Amount": 5000.0},
        {"Date": "2026-07-01", "Type": "Income", "Category": "Sales", "Description": "Sales", "Amount": 10000.0},
        {"Date": "2026-07-02", "Type": "Expense", "Category": "Rent", "Description": "Rent", "Amount": 5000.0},
    ]
    df_flat = pd.DataFrame(flat_data)
    df_flat["Date"] = pd.to_datetime(df_flat["Date"])
    fc_flat = generate_short_term_forecast(df_flat)
    risk_flat = detect_financial_risks(df_flat, fc_flat)

    assert fc_flat["next_month"]["projected_revenue"] == 10000.0, "Flat data should project constant revenue"
    assert risk_flat["revenue_risk"]["severity"] == "Low"
    assert risk_flat["expense_risk"]["severity"] == "Low"
    print("[PASS] Passed: Flat financial data test")

    # 8. Test Executive Rule-Based Fallback Explanation
    explanation = get_risk_forecast_fallback_explanation(fc_inc, risk_inc)
    assert "Short-Term Financial Outlook" in explanation
    assert "Revenue Risk" in explanation
    print("[PASS] Passed: Executive fallback explanation test")

    print("\n=== ALL RISK & FORECAST TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_tests()
