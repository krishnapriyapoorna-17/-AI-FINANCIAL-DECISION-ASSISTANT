"""
health_score.py
Will calculate overall financial health scores, risk assessment, and scoring logic.
"""


def calculate_health_score(summary, monthly_df, expense_flags):
    """
    Takes:
      - summary: the dictionary from calculate_summary() (total_revenue, total_expenses,
        net_profit, profit_margin, net_cash_flow)
      - monthly_df: the DataFrame from monthly_summary() (Month, Income, Expense, Net)
      - expense_flags: the list from detect_expense_flags()

    Computes a rule-based 0-100 score from four weighted components, each scored 0-100
    individually, then combined with these weights:
      - Profit Margin Score (weight 40%):
          profit_margin >= 20 -> 100
          profit_margin >= 10 -> 75
          profit_margin >= 0  -> 50
          profit_margin < 0   -> 20
      - Expense Ratio Score (weight 25%):
          expense_ratio = total_expenses / total_revenue if total_revenue > 0 else 1.0 (worst case)
          expense_ratio <= 0.6 -> 100
          expense_ratio <= 0.8 -> 75
          expense_ratio <= 1.0 -> 50
          expense_ratio > 1.0  -> 20
      - Cash Flow Score (weight 20%):
          Based on how many of the last 3 months (or fewer if less data available) in
          monthly_df had Net >= 0.
          all positive -> 100
          majority positive -> 70
          majority negative -> 40
          all negative -> 20
          (if fewer than 1 month of data, default to 50 - neutral/unknown)
      - Expense Growth Score (weight 15%):
          Based on len(expense_flags):
          0 flags -> 100
          1 flag  -> 70
          2 flags -> 40
          3+ flags -> 20

    Returns a dictionary:
      - overall_score: int (weighted sum, rounded to nearest whole number, clamped 0-100)
      - rating: str, one of "Excellent" (80-100), "Good" (60-79), "Fair" (40-59), "Poor" (0-39)
      - components: dict with keys profit_margin_score, expense_ratio_score, cash_flow_score,
        expense_growth_score, each holding {score: int, weight_pct: int, explanation: str}
    """
    # Step 1: Default fallbacks for safe error handling
    if summary is None:
        summary = {"total_revenue": 0.0, "total_expenses": 0.0, "profit_margin": 0.0}
    if expense_flags is None:
        expense_flags = []

    # Component 1: Profit Margin Score (40% weight)
    profit_margin = float(summary.get("profit_margin", 0.0))
    if profit_margin >= 20.0:
        pm_score = 100
        pm_exp = f"Profit margin of {profit_margin:.1f}% exceeds the 20.0% benchmark for strong financial health."
    elif profit_margin >= 10.0:
        pm_score = 75
        pm_exp = f"Profit margin of {profit_margin:.1f}% falls in the healthy range (10.0% - 20.0%)."
    elif profit_margin >= 0.0:
        pm_score = 50
        pm_exp = f"Profit margin of {profit_margin:.1f}% is breaking even (0.0% - 10.0%)."
    else:
        pm_score = 20
        pm_exp = f"Profit margin of {profit_margin:.1f}% indicates operating at a net loss."

    # Component 2: Expense Ratio Score (25% weight)
    revenue = float(summary.get("total_revenue", 0.0))
    expenses = float(summary.get("total_expenses", 0.0))

    # Avoid divide-by-zero: default to 1.0 (worst case) if revenue is 0
    if revenue > 0:
        expense_ratio = expenses / revenue
    else:
        expense_ratio = 1.0

    exp_ratio_pct = expense_ratio * 100.0

    if expense_ratio <= 0.6:
        er_score = 100
        er_exp = f"Expense ratio of {exp_ratio_pct:.1f}% is low and optimal (<= 60% of revenue)."
    elif expense_ratio <= 0.8:
        er_score = 75
        er_exp = f"Expense ratio of {exp_ratio_pct:.1f}% is in the healthy range (60% - 80% of revenue)."
    elif expense_ratio <= 1.0:
        er_score = 50
        er_exp = f"Expense ratio of {exp_ratio_pct:.1f}% consumes most revenue (80% - 100%)."
    else:
        er_score = 20
        er_exp = f"Expense ratio of {exp_ratio_pct:.1f}% exceeds total revenue (> 100%)."

    # Component 3: Cash Flow Score (20% weight)
    if monthly_df is None or monthly_df.empty or "Net" not in monthly_df.columns:
        cf_score = 50
        cf_exp = "Insufficient monthly data to evaluate cash flow trend."
    else:
        recent_months = monthly_df.tail(3)
        total_months = len(recent_months)
        if total_months < 1:
            cf_score = 50
            cf_exp = "Insufficient monthly data to evaluate cash flow trend."
        else:
            pos_months = (recent_months["Net"] >= 0).sum()
            if pos_months == total_months:
                cf_score = 100
                cf_exp = f"All recent months ({pos_months} of {total_months}) maintained positive net cash flow."
            elif pos_months > (total_months / 2.0):
                cf_score = 70
                cf_exp = f"Majority of recent months ({pos_months} of {total_months}) had positive net cash flow."
            elif pos_months > 0:
                cf_score = 40
                cf_exp = f"Minority of recent months ({pos_months} of {total_months}) had positive net cash flow."
            else:
                cf_score = 20
                cf_exp = f"All recent months (0 of {total_months}) experienced negative net cash flow."

    # Component 4: Expense Growth Score (15% weight)
    num_flags = len(expense_flags)
    if num_flags == 0:
        eg_score = 100
        eg_exp = "No unusual month-over-month expense category spikes detected."
    elif num_flags == 1:
        eg_score = 70
        eg_exp = f"1 expense category flagged for an unusual month-over-month increase (>20%)."
    elif num_flags == 2:
        eg_score = 40
        eg_exp = f"2 expense categories flagged for unusual month-over-month increases (>20%)."
    else:
        eg_score = 20
        eg_exp = f"{num_flags} expense categories flagged for unusual month-over-month increases (>20%)."

    # Step 2: Weighted Sum Calculation
    weighted_sum = (
        (pm_score * 0.40) +
        (er_score * 0.25) +
        (cf_score * 0.20) +
        (eg_score * 0.15)
    )
    overall_score = min(100, max(0, int(round(weighted_sum))))

    # Step 3: Determine Rating Category
    if overall_score >= 80:
        rating = "Excellent"
    elif overall_score >= 60:
        rating = "Good"
    elif overall_score >= 40:
        rating = "Fair"
    else:
        rating = "Poor"

    components = {
        "profit_margin_score": {
            "score": pm_score,
            "weight_pct": 40,
            "explanation": pm_exp,
        },
        "expense_ratio_score": {
            "score": er_score,
            "weight_pct": 25,
            "explanation": er_exp,
        },
        "cash_flow_score": {
            "score": cf_score,
            "weight_pct": 20,
            "explanation": cf_exp,
        },
        "expense_growth_score": {
            "score": eg_score,
            "weight_pct": 15,
            "explanation": eg_exp,
        },
    }

    return {
        "overall_score": overall_score,
        "rating": rating,
        "components": components,
    }
