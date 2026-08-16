"""
anomaly_detection.py
Module for statistical transaction anomaly detection, category spending spike analysis,
trend deviation checks, and AI/rule-based plain business explanations.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv


def detect_transaction_anomalies(df, z_medium=2.5, z_high=3.0, min_samples=3):
    """
    Detects unusual individual transaction amounts grouped by expense category using Z-score.

    Parameters:
      df: Cleaned transactions DataFrame (Date, Type, Category, Description, Amount)
      z_medium: Float threshold for Medium severity (default 2.5)
      z_high: Float threshold for High severity (default 3.0)
      min_samples: Minimum number of transactions required in a category to compute Z-score (default 3)

    Returns:
      List of dicts representing flagged transaction anomalies sorted by absolute Z-score descending.
    """
    if df is None or df.empty or "Type" not in df.columns or "Amount" not in df.columns or "Category" not in df.columns:
        return []

    # Step 1: Filter to Expense transactions (case-normalized)
    df_temp = df.copy()
    df_temp["Type_Norm"] = df_temp["Type"].astype(str).str.strip().str.title()
    exp_df = df_temp[df_temp["Type_Norm"] == "Expense"].copy()

    if exp_df.empty or len(exp_df) < min_samples:
        return []

    # Ensure numeric amount and clean date
    exp_df["Amount"] = pd.to_numeric(exp_df["Amount"], errors="coerce")
    exp_df = exp_df.dropna(subset=["Amount", "Category"])

    if exp_df.empty:
        return []

    anomalies = []

    # Step 2: Group by Category and compute Z-score per transaction
    grouped = exp_df.groupby("Category")

    for category, group in grouped:
        # Skip categories with insufficient data to prevent false positives
        if len(group) < min_samples:
            continue

        mean_val = float(group["Amount"].mean())
        std_val = float(group["Amount"].std(ddof=1)) if len(group) > 1 else 0.0

        # Skip if standard deviation is 0 (all identical amounts) or NaN
        if std_val == 0.0 or np.isnan(std_val):
            continue

        for idx, row in group.iterrows():
            amt = float(row["Amount"])
            z_score = (amt - mean_val) / std_val
            abs_z = abs(z_score)

            if abs_z >= z_medium:
                severity = "High" if abs_z >= z_high else "Medium"
                diff = amt - mean_val
                direction = "above" if diff > 0 else "below"

                # Standardized date formatting for UI presentation
                date_val = row["Date"]
                date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)

                reason = (
                    f"Transaction amount of ${amt:,.2f} is ${abs(diff):,.2f} {direction} "
                    f"the category average of ${mean_val:,.2f} (Z-score: {z_score:+.2f}). Requires review."
                )

                anomalies.append({
                    "Date": date_str,
                    "Description": str(row.get("Description", "N/A")),
                    "Category": str(category),
                    "Amount": round(amt, 2),
                    "Expected/Average": round(mean_val, 2),
                    "Z-score": round(z_score, 2),
                    "Severity": severity,
                    "Reason": reason,
                })

    # Sort descending by absolute Z-score
    anomalies.sort(key=lambda x: abs(x["Z-score"]), reverse=True)
    return anomalies


def detect_category_anomalies(df, threshold_pct=30.0):
    """
    Identifies expense categories whose current period spending is unusually high
    compared with their historical monthly baseline.

    Returns list of dicts with:
      Category, Current Amount, Average Amount, Difference, Percentage Difference, Severity, Reason
    """
    if df is None or df.empty or "Type" not in df.columns or "Date" not in df.columns:
        return []

    df_temp = df.copy()
    df_temp["Type_Norm"] = df_temp["Type"].astype(str).str.strip().str.title()
    exp_df = df_temp[df_temp["Type_Norm"] == "Expense"].copy()

    if exp_df.empty:
        return []

    exp_df["Date"] = pd.to_datetime(exp_df["Date"], errors="coerce")
    exp_df = exp_df.dropna(subset=["Date", "Amount"])

    if exp_df.empty:
        return []

    exp_df["Month"] = exp_df["Date"].dt.strftime("%Y-%m")
    pivoted = exp_df.groupby(["Month", "Category"])["Amount"].sum().unstack(fill_value=0.0)

    if pivoted.empty:
        return []

    months = pivoted.index.tolist()
    cat_anomalies = []

    if len(months) >= 2:
        # Latest month vs prior months average
        latest_month = months[-1]
        prior_months_df = pivoted.iloc[:-1]
        latest_row = pivoted.loc[latest_month]

        for category in pivoted.columns:
            hist_avg = float(prior_months_df[category].mean())
            curr_val = float(latest_row[category])

            if hist_avg > 0:
                pct_diff = ((curr_val - hist_avg) / hist_avg) * 100.0
                diff = curr_val - hist_avg

                if pct_diff >= threshold_pct:
                    severity = "High" if pct_diff >= 50.0 else "Medium"
                    reason = (
                        f"Current month spending (${curr_val:,.2f}) for '{category}' is "
                        f"{pct_diff:.1f}% higher than historical monthly average (${hist_avg:,.2f})."
                    )
                    cat_anomalies.append({
                        "Category": str(category),
                        "Current Amount": round(curr_val, 2),
                        "Average Amount": round(hist_avg, 2),
                        "Difference": round(diff, 2),
                        "Percentage Difference": round(pct_diff, 1),
                        "Severity": severity,
                        "Reason": reason,
                    })
    else:
        # Single month dataset fallback: compare category total against overall average expense category
        cat_totals = exp_df.groupby("Category")["Amount"].sum()
        avg_cat_total = cat_totals.mean()
        std_cat_total = cat_totals.std()

        if std_cat_total and std_cat_total > 0:
            for category, total_val in cat_totals.items():
                z_cat = (total_val - avg_cat_total) / std_cat_total
                if z_cat >= 1.5:
                    pct_diff = ((total_val - avg_cat_total) / avg_cat_total) * 100.0 if avg_cat_total > 0 else 0.0
                    severity = "High" if z_cat >= 2.0 else "Medium"
                    reason = (
                        f"Category '{category}' total (${total_val:,.2f}) is significantly above "
                        f"the average category total (${avg_cat_total:,.2f})."
                    )
                    cat_anomalies.append({
                        "Category": str(category),
                        "Current Amount": round(total_val, 2),
                        "Average Amount": round(avg_cat_total, 2),
                        "Difference": round(total_val - avg_cat_total, 2),
                        "Percentage Difference": round(pct_diff, 1),
                        "Severity": severity,
                        "Reason": reason,
                    })

    cat_anomalies.sort(key=lambda x: x["Percentage Difference"], reverse=True)
    return cat_anomalies


def detect_trend_anomalies(df, threshold_pct=25.0):
    """
    Identifies macro month-over-month trend anomalies, such as sudden expense surges
    or revenue declines across consecutive months.
    """
    if df is None or df.empty or "Date" not in df.columns or "Type" not in df.columns:
        return []

    df_temp = df.copy()
    df_temp["Date"] = pd.to_datetime(df_temp["Date"], errors="coerce")
    df_temp["Type_Norm"] = df_temp["Type"].astype(str).str.strip().str.title()
    df_temp = df_temp.dropna(subset=["Date", "Amount"])

    if df_temp.empty:
        return []

    df_temp["Month"] = df_temp["Date"].dt.strftime("%Y-%m")
    monthly = df_temp.groupby(["Month", "Type_Norm"])["Amount"].sum().unstack(fill_value=0.0)

    if len(monthly) < 2:
        return []

    if "Income" not in monthly.columns:
        monthly["Income"] = 0.0
    if "Expense" not in monthly.columns:
        monthly["Expense"] = 0.0

    trend_anomalies = []
    prior_income = float(monthly.iloc[-2]["Income"])
    curr_income = float(monthly.iloc[-1]["Income"])
    prior_expense = float(monthly.iloc[-2]["Expense"])
    curr_expense = float(monthly.iloc[-1]["Expense"])

    latest_month_str = monthly.index[-1]
    prior_month_str = monthly.index[-2]

    # Revenue Drop Detection
    if prior_income > 0:
        inc_pct = ((curr_income - prior_income) / prior_income) * 100.0
        if inc_pct <= -threshold_pct:
            severity = "High" if inc_pct <= -40.0 else "Medium"
            reason = (
                f"Monthly revenue dropped by {abs(inc_pct):.1f}% in {latest_month_str} "
                f"(${curr_income:,.2f}) compared to {prior_month_str} (${prior_income:,.2f})."
            )
            trend_anomalies.append({
                "Period": latest_month_str,
                "Metric": "Revenue Drop",
                "Current Value": round(curr_income, 2),
                "Prior Value": round(prior_income, 2),
                "Percentage Change": round(inc_pct, 1),
                "Severity": severity,
                "Reason": reason,
            })

    # Expense Surge Detection
    if prior_expense > 0:
        exp_pct = ((curr_expense - prior_expense) / prior_expense) * 100.0
        if exp_pct >= threshold_pct:
            severity = "High" if exp_pct >= 40.0 else "Medium"
            reason = (
                f"Total monthly expenses increased by {exp_pct:.1f}% in {latest_month_str} "
                f"(${curr_expense:,.2f}) compared to {prior_month_str} (${prior_expense:,.2f})."
            )
            trend_anomalies.append({
                "Period": latest_month_str,
                "Metric": "Expense Spike",
                "Current Value": round(curr_expense, 2),
                "Prior Value": round(prior_expense, 2),
                "Percentage Change": round(exp_pct, 1),
                "Severity": severity,
                "Reason": reason,
            })

    return trend_anomalies


def get_anomaly_fallback_explanation(tx_anomalies, cat_anomalies, trend_anomalies):
    """
    Rule-based deterministic summary explanation generator when AI is unavailable.
    Formats executive business language explanations.
    """
    total_anomalies = len(tx_anomalies) + len(cat_anomalies) + len(trend_anomalies)

    if total_anomalies == 0:
        return (
            "### What Needs Attention?\n\n"
            "**No unusual financial activities detected.** All transaction amounts, category spending levels, "
            "and month-over-month trends appear consistent with historical baselines."
        )

    lines = ["### What Needs Attention? (Executive Summary)\n"]

    high_tx = [a for a in tx_anomalies if a["Severity"] == "High"]
    med_tx = [a for a in tx_anomalies if a["Severity"] == "Medium"]

    if high_tx:
        lines.append(f"• **High Priority Transaction Review**: Found {len(high_tx)} high-variance transaction(s) requiring immediate attention:")
        for item in high_tx[:3]:
            lines.append(f"  - **{item['Category']}** ({item['Date']}): ${item['Amount']:,.2f} vs category average ${item['Expected/Average']:,.2f}. *{item['Description']}*")
    elif med_tx:
        lines.append(f"• **Moderate Transaction Review**: Found {len(med_tx)} moderate-variance transaction(s):")
        for item in med_tx[:3]:
            lines.append(f"  - **{item['Category']}** ({item['Date']}): ${item['Amount']:,.2f} vs category average ${item['Expected/Average']:,.2f}.")

    if cat_anomalies:
        lines.append("\n• **Category Spending Spikes**:")
        for cat in cat_anomalies[:2]:
            lines.append(f"  - **{cat['Category']}**: ${cat['Current Amount']:,.2f} (+{cat['Percentage Difference']:.1f}% over baseline ${cat['Average Amount']:,.2f}).")

    if trend_anomalies:
        lines.append("\n• **Macro Trend Alerts**:")
        for tr in trend_anomalies:
            lines.append(f"  - **{tr['Metric']}** ({tr['Period']}): {tr['Reason']}")

    lines.append("\n*Recommendation: Verify flagged transactions for unusual vendor charges, billing entry mistakes, or unbudgeted operational expenses.*")

    return "\n".join(lines)


def get_anomaly_ai_explanation(tx_anomalies, cat_anomalies, trend_anomalies):
    """
    Invokes Google Gemini API with summary metadata of detected anomalies to generate
    plain business language explanations. Falls back cleanly to rule-based explanation on error.
    """
    total_anomalies = len(tx_anomalies) + len(cat_anomalies) + len(trend_anomalies)
    if total_anomalies == 0:
        return get_anomaly_fallback_explanation(tx_anomalies, cat_anomalies, trend_anomalies)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    if not api_key or api_key.strip() == "" or api_key.strip() == "PASTE_KEY_HERE":
        return get_anomaly_fallback_explanation(tx_anomalies, cat_anomalies, trend_anomalies)

    try:
        from google import genai

        prompt = f"""
You are an expert small business financial analyst. Synthesize the following detected financial anomalies into a clear, executive "What Needs Attention?" summary for a business owner.

### DETECTED TRANSACTION ANOMALIES ({len(tx_anomalies)} items):
"""
        for a in tx_anomalies[:5]:
            prompt += f"- [{a['Severity']}] {a['Category']} on {a['Date']}: ${a['Amount']:,.2f} (Category Average: ${a['Expected/Average']:,.2f}, Z-score: {a['Z-score']}). Description: {a['Description']}\n"

        prompt += f"\n### CATEGORY SPENDING SPIKES ({len(cat_anomalies)} items):\n"
        for c in cat_anomalies[:3]:
            prompt += f"- [{c['Severity']}] {c['Category']}: ${c['Current Amount']:,.2f} vs average ${c['Average Amount']:,.2f} (+{c['Percentage Difference']:.1f}%)\n"

        prompt += f"\n### TREND ALERTS ({len(trend_anomalies)} items):\n"
        for t in trend_anomalies:
            prompt += f"- [{t['Severity']}] {t['Metric']}: {t['Reason']}\n"

        prompt += """
INSTRUCTIONS:
1. Write a 2-3 sentence executive overview explaining the top items that need review.
2. Group key observations into clear bullet points using plain business terms.
3. Do NOT use the word "fraud". Use terms like "unusual", "potential anomaly", "variance", or "requires review".
4. Provide 2-3 practical, actionable next steps (e.g. audit invoice, check vendor contract).
Keep the tone professional, objective, and clear.
"""
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model=model_name.strip(),
            contents=prompt,
        )

        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return get_anomaly_fallback_explanation(tx_anomalies, cat_anomalies, trend_anomalies)

    except Exception:
        return get_anomaly_fallback_explanation(tx_anomalies, cat_anomalies, trend_anomalies)
