"""
risk_forecast.py
Module for trend-based short-term financial forecasting, multi-factor risk detection,
and AI/rule-based risk explanation generation.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from modules.financial_analysis import monthly_summary


def _add_months(year_month_str, num_months):
    """Utility to add months to a 'YYYY-MM' string."""
    dt = datetime.strptime(year_month_str, "%Y-%m")
    month = dt.month - 1 + num_months
    year = dt.year + month // 12
    month = month % 12 + 1
    return f"{year:04d}-{month:02d}"


def generate_short_term_forecast(df, forecast_months=2, min_periods=2):
    """
    Generates a simple, explainable trend-based forecast for Revenue, Expenses, Net Profit,
    and Net Cash Flow for the next 1-2 months using historical monthly aggregations.

    Returns:
      Dict with:
        has_sufficient_data: bool
        message: str (if insufficient data)
        historical: DataFrame (Month, Income, Expense, Net)
        forecast: DataFrame (Month, Income, Expense, Net, Type='Projected')
        next_month: Dict with projected next month values
    """
    m_df = monthly_summary(df)

    if m_df is None or m_df.empty or len(m_df) < min_periods:
        current_count = len(m_df) if m_df is not None else 0
        return {
            "has_sufficient_data": False,
            "message": f"Not enough historical data for a reliable forecast (minimum {min_periods} historical months required, currently {current_count} available).",
            "historical": m_df if m_df is not None else pd.DataFrame(),
            "forecast": pd.DataFrame(),
            "next_month": {},
        }

    # Prepare historical data
    hist_df = m_df.copy()
    hist_df["Type"] = "Historical"
    n_hist = len(hist_df)
    x_hist = np.arange(n_hist)

    inc_vals = hist_df["Income"].values.astype(float)
    exp_vals = hist_df["Expense"].values.astype(float)

    # Linear trend slope & intercept calculation
    if n_hist > 1:
        inc_slope, inc_intercept = np.polyfit(x_hist, inc_vals, 1)
        exp_slope, exp_intercept = np.polyfit(x_hist, exp_vals, 1)
    else:
        inc_slope, inc_intercept = 0.0, inc_vals[0]
        exp_slope, exp_intercept = 0.0, exp_vals[0]

    last_month_str = hist_df["Month"].iloc[-1]
    forecast_rows = []

    for i in range(1, forecast_months + 1):
        x_proj = n_hist - 1 + i
        proj_inc = max(0.0, float(inc_slope * x_proj + inc_intercept))
        proj_exp = max(0.0, float(exp_slope * x_proj + exp_intercept))
        proj_net = proj_inc - proj_exp
        proj_month = _add_months(last_month_str, i)

        forecast_rows.append({
            "Month": proj_month,
            "Income": round(proj_inc, 2),
            "Expense": round(proj_exp, 2),
            "Net": round(proj_net, 2),
            "Type": "Projected",
        })

    forecast_df = pd.DataFrame(forecast_rows)
    next_month_data = forecast_rows[0] if forecast_rows else {}

    return {
        "has_sufficient_data": True,
        "historical": hist_df,
        "forecast": forecast_df,
        "next_month": {
            "Month": next_month_data.get("Month", ""),
            "projected_revenue": next_month_data.get("Income", 0.0),
            "projected_expenses": next_month_data.get("Expense", 0.0),
            "projected_net_profit": next_month_data.get("Net", 0.0),
            "projected_cash_flow": next_month_data.get("Net", 0.0),
        },
    }


def detect_financial_risks(df, forecast_res, anomalies=None):
    """
    Evaluates financial risk indicators across 4 core areas:
    - Revenue Risk
    - Expense Risk
    - Profitability Risk
    - Cash Flow Risk

    Returns:
      Dict with risk assessments per category (severity: Low/Medium/High, reason: str)
      and main_risk highlight.
    """
    if not forecast_res or not forecast_res.get("has_sufficient_data"):
        return {
            "has_data": False,
            "revenue_risk": {"severity": "Low", "reason": "Insufficient historical data to evaluate revenue risk."},
            "expense_risk": {"severity": "Low", "reason": "Insufficient historical data to evaluate expense risk."},
            "profitability_risk": {"severity": "Low", "reason": "Insufficient historical data to evaluate profitability risk."},
            "cash_flow_risk": {"severity": "Low", "reason": "Insufficient historical data to evaluate cash flow risk."},
            "main_risk": "Insufficient historical data for risk evaluation.",
        }

    hist_df = forecast_res["historical"]
    forecast_df = forecast_res["forecast"]
    next_month = forecast_res["next_month"]

    curr_rev = float(hist_df["Income"].iloc[-1])
    prior_rev = float(hist_df["Income"].iloc[-2]) if len(hist_df) >= 2 else curr_rev
    proj_rev = float(next_month.get("projected_revenue", 0.0))

    curr_exp = float(hist_df["Expense"].iloc[-1])
    prior_exp = float(hist_df["Expense"].iloc[-2]) if len(hist_df) >= 2 else curr_exp
    proj_exp = float(next_month.get("projected_expenses", 0.0))

    curr_net = float(hist_df["Net"].iloc[-1])
    proj_net = float(next_month.get("projected_net_profit", 0.0))

    # 1. REVENUE RISK EVALUATION
    rev_change_pct = ((curr_rev - prior_rev) / prior_rev * 100.0) if prior_rev > 0 else 0.0
    proj_rev_change = ((proj_rev - curr_rev) / curr_rev * 100.0) if curr_rev > 0 else 0.0

    if rev_change_pct <= -20.0 or proj_rev_change <= -20.0:
        rev_sev = "High"
        rev_reason = f"Revenue shows significant contraction ({rev_change_pct:.1f}% recent drop, projected to reach ${proj_rev:,.2f})."
    elif rev_change_pct < -5.0 or proj_rev_change < -5.0:
        rev_sev = "Medium"
        rev_reason = f"Revenue shows moderate decline ({rev_change_pct:.1f}% recent drop). Monitor sales pipelines."
    else:
        rev_sev = "Low"
        rev_reason = f"Revenue trend is stable or growing (recent period ${curr_rev:,.2f})."

    # 2. EXPENSE RISK EVALUATION
    exp_change_pct = ((curr_exp - prior_exp) / prior_exp * 100.0) if prior_exp > 0 else 0.0
    proj_exp_change = ((proj_exp - curr_exp) / curr_exp * 100.0) if curr_exp > 0 else 0.0

    # Incorporate Phase 1 expense anomaly findings if available
    anomaly_note = ""
    if anomalies:
        tx_anom, cat_anom, trend_anom = anomalies
        if len(cat_anom) > 0:
            top_spike = cat_anom[0]
            anomaly_note = f" Flagged spending spike in '{top_spike['Category']}' (+{top_spike['Percentage Difference']:.1f}%)."

    if (exp_change_pct >= 25.0 or proj_exp_change >= 25.0) or (exp_change_pct > rev_change_pct + 20.0):
        exp_sev = "High"
        exp_reason = f"Expenses are rising significantly faster than revenue ({exp_change_pct:.1f}% MoM increase to ${curr_exp:,.2f}).{anomaly_note}"
    elif exp_change_pct >= 10.0 or proj_exp_change >= 10.0 or (exp_change_pct > rev_change_pct):
        exp_sev = "Medium"
        exp_reason = f"Expenses increased by {exp_change_pct:.1f}% MoM, outpacing revenue growth rate.{anomaly_note}"
    else:
        exp_sev = "Low"
        exp_reason = f"Expense growth remains controlled and aligned with operational budgets."

    # 3. PROFITABILITY RISK EVALUATION
    curr_margin = (curr_net / curr_rev * 100.0) if curr_rev > 0 else 0.0
    proj_margin = (proj_net / proj_rev * 100.0) if proj_rev > 0 else 0.0

    if proj_net < 0 or curr_net < 0:
        prof_sev = "High"
        prof_reason = f"Operating at a net loss (current net profit: ${curr_net:,.2f}, projected: ${proj_net:,.2f})."
    elif curr_margin < 10.0 or proj_margin < 10.0:
        prof_sev = "Medium"
        prof_reason = f"Profit margin is narrow ({curr_margin:.1f}% current, projected {proj_margin:.1f}%). Vulnerable to cost increases."
    else:
        prof_sev = "Low"
        prof_reason = f"Profitability remains healthy with a solid profit margin of {curr_margin:.1f}%."

    # 4. CASH FLOW RISK EVALUATION
    recent_nets = hist_df["Net"].tail(3).values
    negative_months = (recent_nets < 0).sum()

    if proj_net < 0 or negative_months >= 2:
        cf_sev = "High"
        cf_reason = f"Persistent negative net cash flow ({negative_months} of recent months negative). Cash reserves may be depleted."
    elif negative_months == 1 or proj_net < (curr_rev * 0.05):
        cf_sev = "Medium"
        cf_reason = f"Cash flow buffer is minimal. Next month projected net cash flow is ${proj_net:,.2f}."
    else:
        cf_sev = "Low"
        cf_reason = f"Positive cash flow trajectory maintained across recent periods."

    # Determine Primary Main Risk
    sev_order = {"High": 3, "Medium": 2, "Low": 1}
    risks_list = [
        ("Expense Risk", exp_sev, exp_reason),
        ("Profitability Risk", prof_sev, prof_reason),
        ("Revenue Risk", rev_sev, rev_reason),
        ("Cash Flow Risk", cf_sev, cf_reason),
    ]
    risks_sorted = sorted(risks_list, key=lambda r: sev_order[r[1]], reverse=True)
    top_risk_name, top_risk_sev, top_risk_reason = risks_sorted[0]

    if top_risk_sev == "Low":
        main_risk_summary = "Overall financial risk is low. Maintain current operating controls and monitor monthly metrics."
    else:
        main_risk_summary = f"{top_risk_name} is currently the primary area requiring attention ({top_risk_sev} severity): {top_risk_reason}"

    return {
        "has_data": True,
        "revenue_risk": {"severity": rev_sev, "reason": rev_reason},
        "expense_risk": {"severity": exp_sev, "reason": exp_reason},
        "profitability_risk": {"severity": prof_sev, "reason": prof_reason},
        "cash_flow_risk": {"severity": cf_sev, "reason": cf_reason},
        "main_risk": main_risk_summary,
    }


def get_risk_forecast_fallback_explanation(forecast_res, risk_res):
    """
    Generates rule-based executive risk narrative when AI is unconfigured or unavailable.
    """
    if not forecast_res.get("has_sufficient_data"):
        return (
            "### What Should I Watch?\n\n"
            "**Not enough historical data for a reliable forecast.** "
            "At least 2 historical monthly transaction periods are required to compute projections and trend risks."
        )

    next_month = forecast_res["next_month"]
    lines = ["### What Should I Watch? (Executive Risk Summary)\n"]

    lines.append(f"• **Main Financial Focus**: {risk_res['main_risk']}")

    lines.append("\n• **Short-Term Financial Outlook**:")
    lines.append(f"  - **Projected Revenue**: ${next_month['projected_revenue']:,.2f}")
    lines.append(f"  - **Projected Expenses**: ${next_month['projected_expenses']:,.2f}")
    lines.append(f"  - **Projected Net Profit**: ${next_month['projected_net_profit']:,.2f}")

    lines.append("\n• **Key Risk Indicators**:")
    for risk_key in ["revenue_risk", "expense_risk", "profitability_risk", "cash_flow_risk"]:
        r_name = risk_key.replace("_", " ").title()
        r_info = risk_res[risk_key]
        lines.append(f"  - **{r_name} ({r_info['severity']})**: {r_info['reason']}")

    lines.append("\n*Action Recommendation: Monitor weekly cash disbursements and ensure customer invoices are collected promptly.*")

    return "\n".join(lines)


def get_risk_forecast_ai_explanation(forecast_res, risk_res):
    """
    Invokes Google Gemini API with summary forecast and risk metrics to generate plain
    business language guidance. Falls back to rule-based explanation on error.
    """
    if not forecast_res.get("has_sufficient_data"):
        return get_risk_forecast_fallback_explanation(forecast_res, risk_res)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    if not api_key or api_key.strip() == "" or api_key.strip() == "PASTE_KEY_HERE":
        return get_risk_forecast_fallback_explanation(forecast_res, risk_res)

    try:
        from google import genai

        next_m = forecast_res["next_month"]
        prompt = f"""
You are an expert small business CFO. Review the following financial forecast and risk assessment summary, then provide clear, executive "What Should I Watch?" guidance for the business owner.

### SHORT-TERM PROJECTIONS ({next_m.get('Month', 'Next Month')}):
- Projected Revenue: ${next_m.get('projected_revenue', 0.0):,.2f}
- Projected Expenses: ${next_m.get('projected_expenses', 0.0):,.2f}
- Projected Net Profit: ${next_m.get('projected_net_profit', 0.0):,.2f}

### RISK ASSESSMENT:
- Revenue Risk: {risk_res['revenue_risk']['severity']} - {risk_res['revenue_risk']['reason']}
- Expense Risk: {risk_res['expense_risk']['severity']} - {risk_res['expense_risk']['reason']}
- Profitability Risk: {risk_res['profitability_risk']['severity']} - {risk_res['profitability_risk']['reason']}
- Cash Flow Risk: {risk_res['cash_flow_risk']['severity']} - {risk_res['cash_flow_risk']['reason']}
- Main Financial Concern: {risk_res['main_risk']}

INSTRUCTIONS:
1. Write a 2-3 sentence executive overview summarizing the business's near-term outlook and main risk.
2. Outline 2-3 concise bullet points under "What Should I Watch?".
3. Provide 2 clear, practical financial recommendations (e.g. cost containment, revenue acceleration).
4. Use simple, non-jargon business language.
"""
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model=model_name.strip(),
            contents=prompt,
        )

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return get_risk_forecast_fallback_explanation(forecast_res, risk_res)

    except Exception:
        return get_risk_forecast_fallback_explanation(forecast_res, risk_res)
