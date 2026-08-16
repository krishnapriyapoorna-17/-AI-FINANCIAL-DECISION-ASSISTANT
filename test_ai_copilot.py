"""
test_ai_copilot.py
Unit test suite for Phase 4 AI Financial Copilot.
"""

import pandas as pd
from modules.ai_copilot import (
    build_financial_context,
    generate_copilot_financial_summary,
    generate_rule_based_copilot_summary,
    answer_copilot_question,
    generate_rule_based_question_answer,
    is_financial_question,
)


def run_tests():
    print("--- Running AI Financial Copilot Test Suite ---")

    # 1. Test Empty Dataset Context
    empty_df = pd.DataFrame()
    empty_summary = {"total_revenue": 0.0, "total_expenses": 0.0, "net_profit": 0.0, "profit_margin": 0.0}
    empty_ctx = build_financial_context(empty_df, empty_summary, {}, ([], [], []), {}, {}, {})

    assert empty_ctx["has_data"] is False, "Empty DF should yield has_data=False"
    empty_summary_text = generate_copilot_financial_summary(empty_ctx)
    assert "No transaction data" in empty_summary_text or "No financial" in empty_summary_text
    print("[PASS] Passed: Empty dataset context test")

    # 2. Test Normal Financial Context Building & Summary
    summary = {
        "total_revenue": 100000.0,
        "total_expenses": 70000.0,
        "net_profit": 30000.0,
        "profit_margin": 30.0,
        "net_cash_flow": 30000.0,
    }
    health = {"overall_score": 85, "rating": "Excellent"}
    anomalies = ([], [], [])
    forecast = {
        "has_sufficient_data": True,
        "next_month": {"projected_revenue": 105000.0, "projected_expenses": 71000.0, "projected_net_profit": 34000.0},
    }
    risk = {
        "revenue_risk": {"severity": "Low", "reason": "Stable growth"},
        "expense_risk": {"severity": "Low", "reason": "Controlled expenses"},
        "profitability_risk": {"severity": "Low", "reason": "Healthy margin"},
        "cash_flow_risk": {"severity": "Low", "reason": "Positive cash flow"},
        "main_risk": "Overall financial risk is low.",
    }
    scenario = {
        "has_data": True,
        "baseline": {"revenue": 100000.0, "expenses": 70000.0, "profit": 30000.0, "margin": 30.0},
        "scenario": {"revenue": 110000.0, "expenses": 63000.0, "profit": 47000.0, "margin": 42.7},
        "changes": {"rev_diff": 10000.0, "exp_diff": -7000.0, "profit_diff": 17000.0, "margin_diff_pp": 12.7, "rev_pct": 10.0, "exp_pct": -10.0},
    }

    ctx = build_financial_context(None, summary, health, anomalies, forecast, risk, scenario)
    assert ctx["has_data"] is True
    assert ctx["revenue"] == 100000.0
    assert ctx["profit"] == 30000.0
    assert ctx["health_score"] == 85
    assert ctx["has_scenario"] is True
    print("[PASS] Passed: Normal financial context building test")

    # 3. Test High Anomaly Context Inclusion
    high_anomalies = ([
        {"Date": "2026-06-05", "Description": "Extreme Outlier Device", "Category": "Supplies", "Amount": 5000.0, "Expected/Average": 100.0, "Z-score": 3.5, "Severity": "High", "Reason": "Outlier variance"}
    ], [], [])
    ctx_anom = build_financial_context(None, summary, health, high_anomalies, forecast, risk, scenario)
    assert ctx_anom["total_anomalies"] == 1
    assert ctx_anom["high_severity_anomalies"] == 1
    assert "Extreme Outlier Device" in ctx_anom["summary_text"]
    print("[PASS] Passed: High anomaly summary context test")

    # 4. Test Revenue Decline + Expense Surge Context
    risk_high = {
        "revenue_risk": {"severity": "High", "reason": "Revenue dropped 25%"},
        "expense_risk": {"severity": "High", "reason": "Expenses rose 30%"},
        "profitability_risk": {"severity": "High", "reason": "Operating loss projected"},
        "cash_flow_risk": {"severity": "High", "reason": "Negative cash flow"},
        "main_risk": "Expense Risk is currently the primary area requiring attention (High severity): Expenses are rising significantly faster than revenue.",
    }
    ctx_high = build_financial_context(None, summary, health, anomalies, forecast, risk_high, scenario)
    rule_sum = generate_rule_based_copilot_summary(ctx_high)
    assert "Expense growth" in rule_sum or "primary area" in rule_sum
    assert "fraud" not in rule_sum.lower()
    print("[PASS] Passed: High risk revenue decline + expense surge test")

    # 5. Test Rule-Based Fallback Summary & Recommendations
    fallback_sum = generate_rule_based_copilot_summary(ctx)
    assert "AI Financial Summary" in fallback_sum
    assert "AI Recommendations" in fallback_sum
    assert "$100,000.00" in fallback_sum
    print("[PASS] Passed: Rule-based fallback summary & recommendations test")

    # 6. Test Scope Validation (Financial vs Out-of-Scope Questions)
    assert is_financial_question("What should I look at first?") is True
    assert is_financial_question("Why is my profit decreasing?") is True
    assert is_financial_question("Which expense area needs attention?") is True
    assert is_financial_question("What is causing my financial risk?") is True
    assert is_financial_question("How can I improve profitability?") is True
    assert is_financial_question("What is the recipe for chocolate cake?") is False
    assert is_financial_question("Who won the football match?") is False
    print("[PASS] Passed: Scope validation test (Financial vs Out-of-Scope)")

    # 7. Test Out-of-Scope Question Response
    out_scope_ans = answer_copilot_question(ctx, "What is the capital of France?")
    assert "don't have enough information to answer that question" in out_scope_ans
    print("[PASS] Passed: Out-of-scope question response test")

    # 8. Test Core Financial Question Answering (Offline Fallback)
    q1_ans = answer_copilot_question(ctx, "What should I look at first?")
    assert "What to look at first" in q1_ans

    q2_ans = answer_copilot_question(ctx, "Why is my profit decreasing?")
    assert "Profitability Analysis" in q2_ans
    assert "$30,000.00" in q2_ans

    q3_ans = answer_copilot_question(ctx, "Which expense area needs attention?")
    assert "Expense Breakdown Analysis" in q3_ans

    q4_ans = answer_copilot_question(ctx, "What is causing my financial risk?")
    assert "Risk Factor Analysis" in q4_ans

    print("[PASS] Passed: Core financial question answering test")

    # 9. Test What-If Scenario Question Answering
    q_scen_ans = answer_copilot_question(ctx, "Explain the What-If scenario results")
    assert "Active Scenario Analysis" in q_scen_ans
    assert "$47,000.00" in q_scen_ans
    print("[PASS] Passed: What-If scenario question answering test")

    # 10. Test No Raw CSV Data Dump / Safety Rules
    summary_text = ctx["summary_text"]
    assert "Date,Type,Category" not in summary_text, "Raw CSV headers must not be in context"
    assert "PASTE_KEY_HERE" not in summary_text
    print("[PASS] Passed: Safety & privacy data packaging test")

    print("\n=== ALL AI FINANCIAL COPILOT TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_tests()
