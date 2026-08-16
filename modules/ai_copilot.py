"""
ai_copilot.py
Module for AI Financial Copilot compact context compilation, executive summary generation,
recommendations engine, question-answering, and rule-based fallback responses.
"""

import os
import re
from dotenv import load_dotenv


def build_financial_context(df, summary, health, anomalies, forecast, risk, scenario):
    """
    Compiles a compact, structured text and dictionary context summarizing overall financial baseline,
    health ratings, anomalies, risk levels, short-term projections, and What-If scenario results.
    Excludes raw transaction CSV logs.
    """
    if summary is None or (summary.get("total_revenue", 0.0) == 0.0 and summary.get("total_expenses", 0.0) == 0.0):
        return {
            "has_data": False,
            "summary_text": "No financial transaction data is currently loaded.",
            "metrics": {},
        }

    # Extract Baseline Metrics
    rev = float(summary.get("total_revenue", 0.0))
    exp = float(summary.get("total_expenses", 0.0))
    profit = float(summary.get("net_profit", 0.0))
    margin = float(summary.get("profit_margin", 0.0))
    cf = float(summary.get("net_cash_flow", profit))

    score = health.get("overall_score", 0) if health else 0
    rating = health.get("rating", "Unknown") if health else "Unknown"

    # Extract Anomalies (Phase 1)
    tx_anom, cat_anom, trend_anom = anomalies if anomalies else ([], [], [])
    tot_anom = len(tx_anom) + len(cat_anom) + len(trend_anom)
    high_anom = len([a for a in tx_anom if a.get("Severity") == "High"]) + len([c for c in cat_anom if c.get("Severity") == "High"])
    top_tx_reasons = [f"- {a['Category']} ({a.get('Description', 'N/A')}): ${a['Amount']:,.2f} - {a['Reason']}" for a in tx_anom[:3]]

    # Extract Risk & Forecast (Phase 2)
    has_fc = forecast.get("has_sufficient_data", False) if forecast else False
    next_m = forecast.get("next_month", {}) if has_fc else {}
    proj_rev = float(next_m.get("projected_revenue", 0.0))
    proj_exp = float(next_m.get("projected_expenses", 0.0))
    proj_profit = float(next_m.get("projected_net_profit", 0.0))

    rev_risk = risk.get("revenue_risk", {}).get("severity", "Low") if risk else "Low"
    exp_risk = risk.get("expense_risk", {}).get("severity", "Low") if risk else "Low"
    prof_risk = risk.get("profitability_risk", {}).get("severity", "Low") if risk else "Low"
    cf_risk = risk.get("cash_flow_risk", {}).get("severity", "Low") if risk else "Low"
    main_concern = risk.get("main_risk", "None") if risk else "None"

    # Extract Scenario (Phase 3)
    has_scen = scenario.get("has_data", False) if scenario else False
    scen_rev_chg = scenario.get("changes", {}).get("rev_pct", 0.0) if has_scen else 0.0
    scen_exp_chg = scenario.get("changes", {}).get("exp_pct", 0.0) if has_scen else 0.0
    scen_profit = scenario.get("scenario", {}).get("profit", 0.0) if has_scen else 0.0
    scen_profit_diff = scenario.get("changes", {}).get("profit_diff", 0.0) if has_scen else 0.0
    scen_margin = scenario.get("scenario", {}).get("margin", 0.0) if has_scen else 0.0
    scen_margin_diff_pp = scenario.get("changes", {}).get("margin_diff_pp", 0.0) if has_scen else 0.0

    context_dict = {
        "has_data": True,
        "revenue": rev,
        "expenses": exp,
        "profit": profit,
        "margin": margin,
        "cash_flow": cf,
        "health_score": score,
        "health_rating": rating,
        "total_anomalies": tot_anom,
        "high_severity_anomalies": high_anom,
        "top_anomaly_reasons": top_tx_reasons,
        "has_forecast": has_fc,
        "proj_revenue": proj_rev,
        "proj_expenses": proj_exp,
        "proj_profit": proj_profit,
        "revenue_risk": rev_risk,
        "expense_risk": exp_risk,
        "profitability_risk": prof_risk,
        "cash_flow_risk": cf_risk,
        "main_concern": main_concern,
        "has_scenario": has_scen,
        "scen_rev_chg": scen_rev_chg,
        "scen_exp_chg": scen_exp_chg,
        "scen_profit": scen_profit,
        "scen_profit_diff": scen_profit_diff,
        "scen_margin": scen_margin,
        "scen_margin_diff_pp": scen_margin_diff_pp,
    }

    # Structured Text Representation for Prompt Inject
    lines = [
        "### COMPACT FINANCIAL SUMMARY CONTEXT:",
        f"- Baseline Metrics: Revenue = ${rev:,.2f}, Expenses = ${exp:,.2f}, Net Profit = ${profit:,.2f}, Profit Margin = {margin:.1f}%, Net Cash Flow = ${cf:,.2f}",
        f"- Health Rating: {score}/100 ({rating})",
        f"- Anomalies (Phase 1): Total = {tot_anom}, High Severity = {high_anom}",
    ]
    if top_tx_reasons:
        lines.append("  Top Flagged Anomaly Items:\n  " + "\n  ".join(top_tx_reasons))

    if has_fc:
        lines.append(f"- Projections (Phase 2): Next Month Revenue = ${proj_rev:,.2f}, Expenses = ${proj_exp:,.2f}, Net Profit = ${proj_profit:,.2f}")
    else:
        lines.append("- Projections (Phase 2): Insufficient historical periods (< 2 months) for projection.")

    lines.append(f"- Risk Indicators (Phase 2): Revenue Risk = {rev_risk}, Expense Risk = {exp_risk}, Profitability Risk = {prof_risk}, Cash Flow Risk = {cf_risk}")
    lines.append(f"- Main Concern: {main_concern}")

    if has_scen:
        lines.append(f"- Active What-If Scenario (Phase 3): Assumptions (Rev {scen_rev_chg:+.1f}%, Exp {scen_exp_chg:+.1f}%) -> Projected Profit = ${scen_profit:,.2f} (Diff = ${scen_profit_diff:+,.2f}), Margin = {scen_margin:.1f}% (Diff = {scen_margin_diff_pp:+.1f} pp)")

    context_dict["summary_text"] = "\n".join(lines)
    return context_dict


def generate_rule_based_copilot_summary(context_dict):
    """
    Generates a deterministic rule-based executive summary and recommendations when AI is unavailable.
    """
    if not context_dict or not context_dict.get("has_data"):
        return (
            "### AI Financial Summary\n\n"
            "**No transaction data available.** Please upload a CSV file or load sample data."
        )

    rev = context_dict["revenue"]
    exp = context_dict["expenses"]
    profit = context_dict["profit"]
    margin = context_dict["margin"]
    anom_count = context_dict["total_anomalies"]
    main_concern = context_dict["main_concern"]
    exp_risk = context_dict["expense_risk"]

    lines = ["### AI Financial Summary (Executive Overview)\n"]

    if profit >= 0:
        lines.append(
            f"The business generated **${rev:,.2f}** in revenue against **${exp:,.2f}** in expenses, "
            f"yielding a net profit of **${profit:,.2f}** (Profit Margin: **{margin:.1f}%**). "
        )
    else:
        lines.append(
            f"The business is operating at a net loss of **-${abs(profit):,.2f}** on **${rev:,.2f}** revenue "
            f"and **${exp:,.2f}** expenses. "
        )

    lines.append(f"**Main Financial Concern**: {main_concern}\n")

    lines.append("### Key Insight")
    if exp_risk == "High":
        lines.append(f"Expense growth is currently the primary area requiring attention because expenses are increasing faster than revenue.\n")
    elif anom_count > 0:
        lines.append(f"{anom_count} financial anomaly/spike alert(s) require review to prevent unexpected cost leakages.\n")
    else:
        lines.append(f"Financial operations maintain a stable baseline. Focus on revenue optimization.\n")

    lines.append("### AI Recommendations (Actionable Steps)")
    recs = []
    if exp_risk in ["High", "Medium"] or exp > (rev * 0.7):
        recs.append("1. **Review Operating Expenses**: Audit recurring vendor subscriptions and utility costs to trim operating expenses.")
    if anom_count > 0:
        recs.append("2. **Investigate Flagged Anomalies**: Check flagged transaction variances in the Anomaly Detection tab to verify billing entry accuracy.")
    if margin < 15.0:
        recs.append("3. **Protect Profit Margins**: Re-evaluate pricing strategies or negotiate volume discounts with suppliers to raise margins above 15%.")
    else:
        recs.append("3. **Reinvest Operational Profits**: Channel excess net profit into high-ROI marketing and revenue generation initiatives.")
    recs.append("4. **Monitor Cash Flow Runway**: Maintain cash reserves to cover at least 3 months of recurring operational expenses.")

    lines.extend(recs)
    return "\n".join(lines)


def generate_copilot_financial_summary(context_dict):
    """
    Calls Google Gemini API using official google-genai SDK to generate executive summary & recommendations.
    Falls back cleanly to rule-based summary on missing API key or error.
    """
    if not context_dict or not context_dict.get("has_data"):
        return generate_rule_based_copilot_summary(context_dict)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    if not api_key or api_key.strip() == "" or api_key.strip() == "PASTE_KEY_HERE":
        return generate_rule_based_copilot_summary(context_dict)

    try:
        from google import genai

        prompt = f"""
You are an expert small business CFO acting as an AI Financial Copilot. Review the following compact financial summary context and generate an executive summary report for the business owner.

{context_dict.get('summary_text', '')}

INSTRUCTIONS:
1. Provide a 2-3 paragraph executive summary explaining the current financial situation, main concern, unusual activity, and key risk.
2. Under "### Key Insight", state the single most important financial focus in 1-2 sentences.
3. Under "### AI Recommendations", provide 3-5 practical, actionable recommendations based ONLY on the supplied context.
4. Never use the word "fraud". Use professional business terms (e.g., "unusual spending", "variance", "requires review").
5. Do NOT invent facts or hallucinate external information. Keep your tone encouraging, objective, and executive.
"""
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model=model_name.strip(),
            contents=prompt,
        )

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return generate_rule_based_copilot_summary(context_dict)

    except Exception:
        return generate_rule_based_copilot_summary(context_dict)


def is_financial_question(question_text):
    """
    Checks if a user question is related to financial analytics, revenue, expenses,
    profit, margins, cash flow, anomalies, risks, forecasts, or What-If scenarios.
    """
    if not question_text or not isinstance(question_text, str):
        return False

    q_lower = question_text.lower().strip()
    keywords = [
        "revenue", "expense", "cost", "profit", "margin", "cash", "flow", "anomaly",
        "anomalies", "risk", "forecast", "projection", "trend", "scenario", "what-if",
        "spending", "first", "look", "decrease", "increase", "attention", "improve",
        "health", "score", "vendor", "payroll", "loss", "budget", "financial", "money",
    ]
    return any(kw in q_lower for kw in keywords)


def generate_rule_based_question_answer(context_dict, question_text):
    """
    Generates a deterministic rule-based response to financial questions when offline or API unavailable.
    """
    if not context_dict or not context_dict.get("has_data"):
        return "No financial dataset is loaded. Please upload a transaction CSV file first."

    q_lower = question_text.lower()
    rev = context_dict["revenue"]
    exp = context_dict["expenses"]
    profit = context_dict["profit"]
    margin = context_dict["margin"]
    tot_anom = context_dict["total_anomalies"]
    main_concern = context_dict["main_concern"]
    exp_risk = context_dict["expense_risk"]

    if "first" in q_lower or "look" in q_lower or "priority" in q_lower:
        if tot_anom > 0:
            return f"**What to look at first**: You should review the **{tot_anom} flagged anomaly item(s)** under the Anomaly Detection tab to verify unusual spending variances before they compound."
        elif exp_risk in ["High", "Medium"]:
            return f"**What to look at first**: Focus on **operating expense growth**. Total expenses (${exp:,.2f}) are consuming {exp/rev*100:.1f}% of revenue."
        else:
            return f"**What to look at first**: Financial metrics are stable. Focus on sales expansion to increase net revenue beyond ${rev:,.2f}."

    elif "profit" in q_lower or "decrease" in q_lower or "margin" in q_lower:
        return f"**Profitability Analysis**: Net profit is currently **${profit:,.2f}** (Profit Margin: **{margin:.1f}%**). Main financial driver: {main_concern}"

    elif "expense" in q_lower or "spend" in q_lower or "cost" in q_lower:
        return f"**Expense Breakdown Analysis**: Total expenses are **${exp:,.2f}**. Expense Risk rating is **{exp_risk}**. {tot_anom} spending spike(s) require review."

    elif "risk" in q_lower or "cause" in q_lower:
        return f"**Risk Factor Analysis**: **{main_concern}**. Risk Ratings: Revenue ({context_dict['revenue_risk']}), Expenses ({context_dict['expense_risk']}), Profitability ({context_dict['profitability_risk']}), Cash Flow ({context_dict['cash_flow_risk']})."

    elif "scenario" in q_lower or "what-if" in q_lower:
        if context_dict.get("has_scenario"):
            return f"**Active Scenario Analysis**: Under current assumptions (Rev {context_dict['scen_rev_chg']:+.1f}%, Exp {context_dict['scen_exp_chg']:+.1f}%), estimated profit reaches **${context_dict['scen_profit']:,.2f}** (Profit Diff: ${context_dict['scen_profit_diff']:+,.2f})."
        else:
            return "No active What-If scenario selected. Visit the What-If Simulator tab to test custom revenue/expense assumptions."

    else:
        return (
            f"**Financial Position Summary**: Based on your dataset, Revenue is **${rev:,.2f}**, Expenses are **${exp:,.2f}**, "
            f"Net Profit is **${profit:,.2f}** ({margin:.1f}% margin), and Health Rating is **{context_dict['health_score']}/100 ({context_dict['health_rating']})**. "
            f"Primary focus: {main_concern}"
        )


def answer_copilot_question(context_dict, user_question):
    """
    Answers user financial queries. First validates topic scope, then calls Gemini API (or rule-based fallback).
    """
    if not user_question or not user_question.strip():
        return "Please ask a question about your business financials."

    # Validate Scope
    if not is_financial_question(user_question):
        return (
            "I can help explain the financial information available in this dashboard, "
            "but I don't have enough information to answer that question."
        )

    if not context_dict or not context_dict.get("has_data"):
        return generate_rule_based_question_answer(context_dict, user_question)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    if not api_key or api_key.strip() == "" or api_key.strip() == "PASTE_KEY_HERE":
        return generate_rule_based_question_answer(context_dict, user_question)

    try:
        from google import genai

        prompt = f"""
You are an AI Financial Copilot for a small business owner. Answer the user's question using ONLY the provided financial summary context.

{context_dict.get('summary_text', '')}

USER QUESTION: "{user_question.strip()}"

INSTRUCTIONS:
1. Answer the question in 2-4 concise, professional business sentences.
2. Use ONLY facts present in the summary context. Do NOT invent transactions or metrics.
3. If the question asks about future scenarios or projections, use clear qualifying words ("Projected", "Under this scenario", "Estimated").
4. Do NOT call anomalies "fraud".
5. Keep your tone clear, helpful, and executive.
"""
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model=model_name.strip(),
            contents=prompt,
        )

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return generate_rule_based_question_answer(context_dict, user_question)

    except Exception:
        return generate_rule_based_question_answer(context_dict, user_question)
