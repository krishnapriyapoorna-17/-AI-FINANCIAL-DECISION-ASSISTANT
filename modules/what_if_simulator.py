"""
what_if_simulator.py
Module for What-If Business Decision Simulation, preset scenarios, custom assumption modeling,
and AI/rule-based business impact interpretation.
"""

import os
from dotenv import load_dotenv


def calculate_scenario(baseline_summary, rev_change_pct, exp_change_pct):
    """
    Computes a hypothetical scenario based on baseline summary metrics and user assumptions.

    Parameters:
      baseline_summary: Dict from calculate_summary() containing total_revenue, total_expenses, net_profit, profit_margin
      rev_change_pct: Float percentage change in revenue (-50 to +50)
      exp_change_pct: Float percentage change in expenses (-50 to +50)

    Returns:
      Dict containing:
        has_data: bool
        baseline: Dict of baseline values
        scenario: Dict of scenario calculated values
        changes: Dict of absolute & percentage point differences
    """
    if not baseline_summary or (baseline_summary.get("total_revenue", 0.0) == 0.0 and baseline_summary.get("total_expenses", 0.0) == 0.0):
        return {
            "has_data": False,
            "baseline": {"revenue": 0.0, "expenses": 0.0, "profit": 0.0, "margin": 0.0, "cash_flow": 0.0},
            "scenario": {"revenue": 0.0, "expenses": 0.0, "profit": 0.0, "margin": 0.0, "cash_flow": 0.0},
            "changes": {"rev_diff": 0.0, "exp_diff": 0.0, "profit_diff": 0.0, "margin_diff_pp": 0.0, "rev_pct": rev_change_pct, "exp_pct": exp_change_pct},
        }

    base_rev = float(baseline_summary.get("total_revenue", 0.0))
    base_exp = float(baseline_summary.get("total_expenses", 0.0))
    base_profit = float(baseline_summary.get("net_profit", 0.0))
    base_margin = float(baseline_summary.get("profit_margin", 0.0))
    base_cf = float(baseline_summary.get("net_cash_flow", base_profit))

    scen_rev = max(0.0, base_rev * (1.0 + float(rev_change_pct) / 100.0))
    scen_exp = max(0.0, base_exp * (1.0 + float(exp_change_pct) / 100.0))
    scen_profit = scen_rev - scen_exp
    scen_margin = (scen_profit / scen_rev * 100.0) if scen_rev > 0.0 else 0.0
    scen_cf = scen_profit

    rev_diff = scen_rev - base_rev
    exp_diff = scen_exp - base_exp
    profit_diff = scen_profit - base_profit
    margin_diff_pp = scen_margin - base_margin

    return {
        "has_data": True,
        "baseline": {
            "revenue": round(base_rev, 2),
            "expenses": round(base_exp, 2),
            "profit": round(base_profit, 2),
            "margin": round(base_margin, 1),
            "cash_flow": round(base_cf, 2),
        },
        "scenario": {
            "revenue": round(scen_rev, 2),
            "expenses": round(scen_exp, 2),
            "profit": round(scen_profit, 2),
            "margin": round(scen_margin, 1),
            "cash_flow": round(scen_cf, 2),
        },
        "changes": {
            "rev_diff": round(rev_diff, 2),
            "exp_diff": round(exp_diff, 2),
            "profit_diff": round(profit_diff, 2),
            "margin_diff_pp": round(margin_diff_pp, 1),
            "rev_pct": float(rev_change_pct),
            "exp_pct": float(exp_change_pct),
        },
    }


def get_preset_scenarios():
    """
    Returns pre-configured simulation scenarios.
    """
    return {
        "Revenue Growth": {"rev_change_pct": 10.0, "exp_change_pct": 0.0, "desc": "10% increase in revenue with unchanged expenses."},
        "Cost Reduction": {"rev_change_pct": 0.0, "exp_change_pct": -10.0, "desc": "10% reduction in operating expenses with unchanged revenue."},
        "Growth + Cost Reduction": {"rev_change_pct": 10.0, "exp_change_pct": -10.0, "desc": "10% revenue increase combined with 10% expense reduction."},
        "Custom Scenario": {"rev_change_pct": 0.0, "exp_change_pct": 0.0, "desc": "Custom user-defined revenue and expense assumptions."},
    }


def generate_business_interpretation(baseline, scenario, changes):
    """
    Produces rule-based business interpretations for a simulated scenario.
    """
    if not baseline or not scenario:
        return "Insufficient baseline data to evaluate business impact."

    scen_profit = scenario.get("profit", 0.0)
    profit_diff = changes.get("profit_diff", 0.0)
    exp_diff = changes.get("exp_diff", 0.0)
    rev_diff = changes.get("rev_diff", 0.0)
    margin_diff_pp = changes.get("margin_diff_pp", 0.0)

    if scen_profit < 0:
        return f"Scenario results in a projected loss of -${abs(scen_profit):,.2f} under these assumptions. Careful cost control or revenue recovery is needed."
    elif profit_diff > 0 and exp_diff < 0 and rev_diff > 0:
        return f"Growth + Cost Reduction scenario significantly improves profitability, adding +${profit_diff:,.2f} to net profit and expanding profit margin by +{margin_diff_pp:.1f} percentage points."
    elif profit_diff > 0 and exp_diff < 0:
        return f"Cost reduction has a positive impact on profitability, expanding profit margin by +{margin_diff_pp:.1f} percentage points."
    elif profit_diff > 0:
        return f"Scenario improves profitability, adding +${profit_diff:,.2f} to estimated net profit."
    elif profit_diff < 0 and rev_diff < 0:
        return f"Lower revenue puts pressure on profitability, reducing net profit by -${abs(profit_diff):,.2f}."
    elif profit_diff < 0:
        return f"Scenario reduces profitability by -${abs(profit_diff):,.2f} and should be reviewed carefully."
    else:
        return "Scenario maintains current baseline financial performance."


def get_simulator_fallback_explanation(baseline, scenario, changes, interpretation):
    """
    Generates rule-based executive scenario summary narrative when AI is unconfigured or unavailable.
    """
    if not baseline or not scenario:
        return "### What Changes?\n\nAdd transaction data to simulate financial scenarios."

    lines = ["### What Changes? (Scenario Impact Breakdown)\n"]

    rev_diff = changes["rev_diff"]
    exp_diff = changes["exp_diff"]
    profit_diff = changes["profit_diff"]
    margin_diff = changes["margin_diff_pp"]

    rev_text = f"increase by +${rev_diff:,.2f}" if rev_diff > 0 else (f"decrease by -${abs(rev_diff):,.2f}" if rev_diff < 0 else "remain unchanged")
    exp_text = f"increase by +${exp_diff:,.2f}" if exp_diff > 0 else (f"decrease by -${abs(exp_diff):,.2f}" if exp_diff < 0 else "remain unchanged")
    profit_text = f"improve by +${profit_diff:,.2f}" if profit_diff > 0 else (f"decline by -${abs(profit_diff):,.2f}" if profit_diff < 0 else "remain unchanged")
    margin_text = f"improve by +{margin_diff:.1f} percentage points" if margin_diff > 0 else (f"decline by {margin_diff:.1f} percentage points" if margin_diff < 0 else "remain unchanged")

    lines.append(f"• **Revenue Impact**: Revenue would {rev_text} (from ${baseline['revenue']:,.2f} to ${scenario['revenue']:,.2f}).")
    lines.append(f"• **Expense Impact**: Expenses would {exp_text} (from ${baseline['expenses']:,.2f} to ${scenario['expenses']:,.2f}).")
    lines.append(f"• **Profit Impact**: Estimated net profit would {profit_text} (from ${baseline['profit']:,.2f} to ${scenario['profit']:,.2f}).")
    lines.append(f"• **Margin Impact**: Profit margin would {margin_text} (from {baseline['margin']:.1f}% to {scenario['margin']:.1f}%).")

    lines.append(f"\n**Business Impact Assessment**:\n\"{interpretation}\"")

    return "\n".join(lines)


def get_simulator_ai_explanation(baseline, scenario, changes, interpretation):
    """
    Invokes Google Gemini API with summary scenario metrics to generate a concise, plain
    business language explanation. Falls back cleanly to rule-based explanation on error.
    """
    if not baseline or not scenario:
        return get_simulator_fallback_explanation(baseline, scenario, changes, interpretation)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    if not api_key or api_key.strip() == "" or api_key.strip() == "PASTE_KEY_HERE":
        return get_simulator_fallback_explanation(baseline, scenario, changes, interpretation)

    try:
        from google import genai

        prompt = f"""
You are an expert small business CFO. Analyze the following hypothetical scenario simulation and provide a clear, 2-3 sentence executive business commentary for the business owner.

### BASELINE POSITION:
- Revenue: ${baseline.get('revenue', 0.0):,.2f}
- Expenses: ${baseline.get('expenses', 0.0):,.2f}
- Net Profit: ${baseline.get('profit', 0.0):,.2f}
- Profit Margin: {baseline.get('margin', 0.0):.1f}%

### SIMULATED ASSUMPTIONS:
- Revenue Change: {changes.get('rev_pct', 0.0):+.1f}%
- Expense Change: {changes.get('exp_pct', 0.0):+.1f}%

### SIMULATED SCENARIO RESULTS:
- Scenario Revenue: ${scenario.get('revenue', 0.0):,.2f} (Diff: ${changes.get('rev_diff', 0.0):+,.2f})
- Scenario Expenses: ${scenario.get('expenses', 0.0):,.2f} (Diff: ${changes.get('exp_diff', 0.0):+,.2f})
- Scenario Net Profit: ${scenario.get('profit', 0.0):,.2f} (Diff: ${changes.get('profit_diff', 0.0):+,.2f})
- Scenario Profit Margin: {scenario.get('margin', 0.0):.1f}% (Diff: {changes.get('margin_diff_pp', 0.0):+.1f} percentage points)

RULE-BASED ASSESSMENT: "{interpretation}"

INSTRUCTIONS:
1. Write a 2-3 sentence executive summary explaining the operational impact of this scenario.
2. Provide 2 bullet points under "What Changes?".
3. State whether this scenario is recommended and what key operational factor to monitor.
4. Keep the tone professional, objective, and clear. Do not guarantee future results.
"""
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model=model_name.strip(),
            contents=prompt,
        )

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return get_simulator_fallback_explanation(baseline, scenario, changes, interpretation)

    except Exception:
        return get_simulator_fallback_explanation(baseline, scenario, changes, interpretation)
