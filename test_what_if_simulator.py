"""
test_what_if_simulator.py
Unit test suite for Phase 3 What-If Business Decision Simulator.
"""

from modules.what_if_simulator import (
    calculate_scenario,
    get_preset_scenarios,
    generate_business_interpretation,
    get_simulator_fallback_explanation,
)


def run_tests():
    print("--- Running What-If Simulator Test Suite ---")

    # 1. Exact Prompt Test Case
    # Current Revenue = 100000, Current Expenses = 70000
    # Revenue Change = +10%, Expense Change = -10%
    base = {
        "total_revenue": 100000.0,
        "total_expenses": 70000.0,
        "net_profit": 30000.0,
        "profit_margin": 30.0,
        "net_cash_flow": 30000.0,
    }

    res = calculate_scenario(base, rev_change_pct=10.0, exp_change_pct=-10.0)

    assert res["has_data"] is True
    assert res["scenario"]["revenue"] == 110000.0, f"Expected 110000, got {res['scenario']['revenue']}"
    assert res["scenario"]["expenses"] == 63000.0, f"Expected 63000, got {res['scenario']['expenses']}"
    assert res["scenario"]["profit"] == 47000.0, f"Expected 47000, got {res['scenario']['profit']}"
    assert res["changes"]["profit_diff"] == 17000.0, f"Expected +17000, got {res['changes']['profit_diff']}"
    assert abs(res["scenario"]["margin"] - 42.7) <= 0.1, f"Expected margin ~42.7%, got {res['scenario']['margin']}"
    assert abs(res["changes"]["margin_diff_pp"] - 12.7) <= 0.1, f"Expected margin diff ~+12.7 pp, got {res['changes']['margin_diff_pp']}"
    print("[PASS] Passed: Exact prompt calculation test (100k Rev, 70k Exp, +10% Rev, -10% Exp -> 47k Profit, +12.7 pp Margin)")

    # 2. Preset Scenarios Test
    presets = get_preset_scenarios()
    assert "Revenue Growth" in presets
    assert "Cost Reduction" in presets
    assert "Growth + Cost Reduction" in presets
    assert presets["Revenue Growth"]["rev_change_pct"] == 10.0
    assert presets["Cost Reduction"]["exp_change_pct"] == -10.0
    print("[PASS] Passed: Preset scenarios configuration test")

    # 3. Test Empty Baseline Dataset
    empty_base = {"total_revenue": 0.0, "total_expenses": 0.0, "net_profit": 0.0, "profit_margin": 0.0}
    res_empty = calculate_scenario(empty_base, rev_change_pct=10.0, exp_change_pct=-10.0)
    assert res_empty["has_data"] is False
    assert res_empty["scenario"]["revenue"] == 0.0
    print("[PASS] Passed: Empty baseline dataset test")

    # 4. Test Revenue Decrease Scenario (-20% Rev, 0% Exp)
    res_rev_dec = calculate_scenario(base, rev_change_pct=-20.0, exp_change_pct=0.0)
    assert res_rev_dec["scenario"]["revenue"] == 80000.0
    assert res_rev_dec["scenario"]["expenses"] == 70000.0
    assert res_rev_dec["scenario"]["profit"] == 10000.0
    assert res_rev_dec["changes"]["profit_diff"] == -20000.0
    interp_dec = generate_business_interpretation(res_rev_dec["baseline"], res_rev_dec["scenario"], res_rev_dec["changes"])
    assert "Lower revenue" in interp_dec or "reduces profitability" in interp_dec
    print("[PASS] Passed: Revenue decrease scenario test")

    # 5. Test Expense Increase Scenario (0% Rev, +20% Exp)
    res_exp_inc = calculate_scenario(base, rev_change_pct=0.0, exp_change_pct=20.0)
    assert res_exp_inc["scenario"]["revenue"] == 100000.0
    assert res_exp_inc["scenario"]["expenses"] == 84000.0
    assert res_exp_inc["scenario"]["profit"] == 16000.0
    assert res_exp_inc["changes"]["profit_diff"] == -14000.0
    print("[PASS] Passed: Expense increase scenario test")

    # 6. Test Projected Loss Scenario (-50% Rev, +20% Exp)
    res_loss = calculate_scenario(base, rev_change_pct=-50.0, exp_change_pct=20.0)
    assert res_loss["scenario"]["revenue"] == 50000.0
    assert res_loss["scenario"]["expenses"] == 84000.0
    assert res_loss["scenario"]["profit"] == -34000.0
    interp_loss = generate_business_interpretation(res_loss["baseline"], res_loss["scenario"], res_loss["changes"])
    assert "loss" in interp_loss.lower()
    print("[PASS] Passed: Projected loss scenario test")

    # 7. Test Zero Baseline Revenue (Expenses only)
    base_zero_rev = {"total_revenue": 0.0, "total_expenses": 5000.0, "net_profit": -5000.0, "profit_margin": 0.0}
    res_zero_rev = calculate_scenario(base_zero_rev, rev_change_pct=10.0, exp_change_pct=-20.0)
    assert res_zero_rev["scenario"]["revenue"] == 0.0
    assert res_zero_rev["scenario"]["expenses"] == 4000.0
    assert res_zero_rev["scenario"]["profit"] == -4000.0
    assert res_zero_rev["scenario"]["margin"] == 0.0, "Margin should be 0.0 when revenue is 0"
    print("[PASS] Passed: Zero baseline revenue edge case test")

    # 8. Test Fallback Summary Explanation Formatting
    explanation = get_simulator_fallback_explanation(res["baseline"], res["scenario"], res["changes"], "Test interpretation")
    assert "What Changes?" in explanation
    assert "Revenue Impact" in explanation
    assert "Expense Impact" in explanation
    assert "Profit Impact" in explanation
    print("[PASS] Passed: Executive fallback explanation test")

    print("\n=== ALL WHAT-IF SIMULATOR TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_tests()
