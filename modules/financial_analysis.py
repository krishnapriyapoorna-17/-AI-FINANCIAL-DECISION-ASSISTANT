"""
financial_analysis.py
Will handle financial metrics computation, cash flow calculation, and trend analysis.
"""

import pandas as pd


def calculate_summary(df):
    """
    Takes the cleaned transactions DataFrame (columns: Date, Type, Category, Description, Amount).
    Type values are either "Income" or "Expense" (case-insensitive, so normalize to title case first).
    Returns a dictionary with:
      - total_revenue: sum of Amount where Type == "Income"
      - total_expenses: sum of Amount where Type == "Expense"
      - net_profit: total_revenue - total_expenses
      - profit_margin: (net_profit / total_revenue * 100) if total_revenue > 0 else 0, rounded to 1 decimal
      - net_cash_flow: same as net_profit for this simple model (revenue in minus expenses out)
    All monetary values rounded to 2 decimals. Handle the case of an empty DataFrame by
    returning all zeros without raising an exception.
    """
    # Step 1: Handle empty DataFrame or missing columns gracefully
    if df is None or df.empty or "Type" not in df.columns or "Amount" not in df.columns:
        return {
            "total_revenue": 0.0,
            "total_expenses": 0.0,
            "net_profit": 0.0,
            "profit_margin": 0.0,
            "net_cash_flow": 0.0,
        }

    # Step 2: Normalize Type column to Title Case for case-insensitive matching
    types_normalized = df["Type"].astype(str).str.strip().str.title()

    # Step 3: Calculate total revenue (Income) and total expenses (Expense)
    total_revenue = float(df[types_normalized == "Income"]["Amount"].sum())
    total_expenses = float(df[types_normalized == "Expense"]["Amount"].sum())

    # Step 4: Calculate net profit, profit margin percentage, and net cash flow
    net_profit = total_revenue - total_expenses
    profit_margin = (net_profit / total_revenue * 100.0) if total_revenue > 0 else 0.0
    net_cash_flow = net_profit

    # Step 5: Round monetary values to 2 decimals and profit margin to 1 decimal
    return {
        "total_revenue": round(total_revenue, 2),
        "total_expenses": round(total_expenses, 2),
        "net_profit": round(net_profit, 2),
        "profit_margin": round(profit_margin, 1),
        "net_cash_flow": round(net_cash_flow, 2),
    }


def monthly_summary(df):
    """
    Takes the cleaned transactions DataFrame.
    Groups by year-month (using the Date column) and Type.
    Returns a pandas DataFrame with columns: Month (as "YYYY-MM" string), Income, Expense, Net
    sorted chronologically. Months with no income or expense should show 0, not NaN.
    Handle empty input DataFrame by returning an empty DataFrame with those columns.
    """
    expected_cols = ["Month", "Income", "Expense", "Net"]

    # Step 1: Handle empty input DataFrame
    if df is None or df.empty or "Date" not in df.columns or "Type" not in df.columns or "Amount" not in df.columns:
        return pd.DataFrame(columns=expected_cols)

    # Step 2: Copy DataFrame, convert Date to datetime, and extract YYYY-MM
    df_temp = df.copy()
    df_temp["Date"] = pd.to_datetime(df_temp["Date"], errors="coerce")
    df_temp = df_temp.dropna(subset=["Date"])

    if df_temp.empty:
        return pd.DataFrame(columns=expected_cols)

    df_temp["Month"] = df_temp["Date"].dt.strftime("%Y-%m")
    df_temp["Type"] = df_temp["Type"].astype(str).str.strip().str.title()

    # Step 3: Group by Month and Type, summing the Amount values
    grouped = df_temp.groupby(["Month", "Type"])["Amount"].sum().unstack(fill_value=0.0)

    # Step 4: Ensure Income and Expense columns exist in grouped output
    if "Income" not in grouped.columns:
        grouped["Income"] = 0.0
    if "Expense" not in grouped.columns:
        grouped["Expense"] = 0.0

    # Step 5: Calculate Net profit per month (Income minus Expense)
    grouped = grouped.reset_index()
    grouped["Net"] = grouped["Income"] - grouped["Expense"]

    # Step 6: Select output columns and sort chronologically by Month
    result_df = grouped[expected_cols].sort_values("Month").reset_index(drop=True)

    return result_df


def expense_by_category(df):
    """
    Takes the cleaned transactions DataFrame.
    Filters to Type == "Expense" (normalize case first).
    Groups by Category, summing Amount.
    Returns a DataFrame with columns: Category, Total, Percentage
    (Percentage = Total / sum of all expenses * 100, rounded to 1 decimal).
    Sorted descending by Total.
    Handle empty/no-expense input by returning an empty DataFrame with those columns.
    """
    expected_cols = ["Category", "Total", "Percentage"]

    # Step 1: Check for empty input or missing required columns
    if df is None or df.empty or "Type" not in df.columns or "Category" not in df.columns or "Amount" not in df.columns:
        return pd.DataFrame(columns=expected_cols)

    # Step 2: Filter to Type == "Expense" (case-insensitive)
    types_norm = df["Type"].astype(str).str.strip().str.title()
    df_expenses = df[types_norm == "Expense"].copy()

    if df_expenses.empty:
        return pd.DataFrame(columns=expected_cols)

    # Step 3: Group by Category and sum Amount
    grouped = df_expenses.groupby("Category")["Amount"].sum().reset_index()
    grouped.columns = ["Category", "Total"]

    # Step 4: Calculate total expenses and percentages
    sum_expenses = grouped["Total"].sum()
    if sum_expenses > 0:
        grouped["Percentage"] = (grouped["Total"] / sum_expenses * 100.0).round(1)
    else:
        grouped["Percentage"] = 0.0

    grouped["Total"] = grouped["Total"].round(2)

    # Step 5: Sort descending by Total
    result_df = grouped.sort_values(by="Total", ascending=False).reset_index(drop=True)

    return result_df[expected_cols]


def monthly_expense_by_category(df):
    """
    Takes the cleaned transactions DataFrame.
    Filters to Type == "Expense".
    Groups by year-month and Category, summing Amount.
    Returns a pivoted DataFrame: rows = Month ("YYYY-MM"), columns = Category, values = total
    expense amount, filled with 0 for missing combinations, sorted chronologically by Month.
    Handle empty input by returning an empty DataFrame.
    """
    if df is None or df.empty or "Type" not in df.columns or "Date" not in df.columns or "Amount" not in df.columns:
        return pd.DataFrame()

    # Step 1: Filter to Type == "Expense"
    df_temp = df.copy()
    types_norm = df_temp["Type"].astype(str).str.strip().str.title()
    df_exp = df_temp[types_norm == "Expense"].copy()

    if df_exp.empty:
        return pd.DataFrame()

    # Step 2: Convert Date to datetime and format as YYYY-MM
    df_exp["Date"] = pd.to_datetime(df_exp["Date"], errors="coerce")
    df_exp = df_exp.dropna(subset=["Date"])

    if df_exp.empty:
        return pd.DataFrame()

    df_exp["Month"] = df_exp["Date"].dt.strftime("%Y-%m")

    # Step 3: Pivot table with Month as index, Category as columns, summing Amount
    pivoted = df_exp.groupby(["Month", "Category"])["Amount"].sum().unstack(fill_value=0.0)

    # Step 4: Sort chronologically by Month
    pivoted = pivoted.sort_index()

    return pivoted


def detect_expense_flags(monthly_cat_df, threshold_pct=20):
    """
    Takes the pivoted monthly_expense_by_category DataFrame.
    For each category, compares the most recent month's value to the prior month's value.
    If the increase is greater than threshold_pct percent (and the prior month's value was
    greater than 0, to avoid divide-by-zero and avoid flagging brand-new categories as
    'increases'), flag it.
    Returns a list of dictionaries, each with:
      - category: str
      - prior_month_amount: float
      - current_month_amount: float
      - pct_change: float (rounded to 1 decimal)
    Sorted descending by pct_change.
    If there are fewer than 2 months of data, return an empty list (not enough data to compare).
    """
    # Step 1: Check if DataFrame has at least 2 rows (2 months) for comparison
    if monthly_cat_df is None or monthly_cat_df.empty or len(monthly_cat_df) < 2:
        return []

    flags = []

    # Step 2: Extract prior month (second-to-last) and current month (last) rows
    prior_row = monthly_cat_df.iloc[-2]
    current_row = monthly_cat_df.iloc[-1]

    # Step 3: Iterate through each category column
    for category in monthly_cat_df.columns:
        prior_val = float(prior_row[category])
        current_val = float(current_row[category])

        # Divide-by-zero handling: Only compute percentage increase if prior_val > 0.
        # This avoids division by zero errors and prevents flagging brand-new expense categories
        # (where prior month was $0) as artificial 100%+ spikes.
        if prior_val > 0:
            pct_change = ((current_val - prior_val) / prior_val) * 100.0

            # Flag if percentage increase exceeds the threshold
            if pct_change > threshold_pct:
                flags.append({
                    "category": str(category),
                    "prior_month_amount": round(prior_val, 2),
                    "current_month_amount": round(current_val, 2),
                    "pct_change": round(pct_change, 1),
                })

    # Step 4: Sort flags descending by pct_change
    flags_sorted = sorted(flags, key=lambda x: x["pct_change"], reverse=True)

    return flags_sorted
