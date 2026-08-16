import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from modules.data_processor import load_sample_data, process_uploaded_file
from modules.financial_analysis import (
    calculate_summary,
    monthly_summary,
    expense_by_category,
    monthly_expense_by_category,
    detect_expense_flags,
)
from modules.health_score import calculate_health_score
from modules.ai_advisor import get_ai_recommendations, get_fallback_recommendations
from modules.anomaly_detection import (
    detect_transaction_anomalies,
    detect_category_anomalies,
    detect_trend_anomalies,
    get_anomaly_ai_explanation,
    get_anomaly_fallback_explanation,
)
from modules.risk_forecast import (
    generate_short_term_forecast,
    detect_financial_risks,
    get_risk_forecast_ai_explanation,
    get_risk_forecast_fallback_explanation,
)
from modules.what_if_simulator import (
    calculate_scenario,
    get_preset_scenarios,
    generate_business_interpretation,
    get_simulator_ai_explanation,
    get_simulator_fallback_explanation,
)
from modules.ai_copilot import (
    build_financial_context,
    generate_copilot_financial_summary,
    generate_rule_based_copilot_summary,
    answer_copilot_question,
)

# Page configuration
st.set_page_config(page_title="AI Financial Decision Assistant", layout="wide")

# Header and Subheader
st.markdown("<h1 style='text-align: center;'>AI Financial Decision Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Analyze your business finances and get AI-powered recommendations</h3>", unsafe_allow_html=True)
st.write("")

# File Uploader Section
uploaded_file = st.file_uploader("Upload your transaction CSV (optional)", type=["csv"])
st.caption("If no file is uploaded, sample demo data will be used.")

# Determine DataFrame to use
df = None

if uploaded_file is not None:
    cleaned_df, msg = process_uploaded_file(uploaded_file)
    if cleaned_df is None:
        st.error(msg)
        df = load_sample_data()
        st.info("Using sample data instead.")
    else:
        df = cleaned_df
        if msg is not None:
            st.warning(msg)
        else:
            st.success("File uploaded and validated successfully.")
else:
    df = load_sample_data()
    st.info("No file uploaded — showing sample data.")

# Store final selected dataframe in session state
st.session_state["transactions_df"] = df

st.write("")

# Sidebar Navigation
st.sidebar.title("Navigation")
navigation = st.sidebar.radio(
    "Select View:",
    ["Financial Dashboard", "Anomaly Detection", "Risk & Forecast", "What-If Simulator", "AI Financial Copilot"]
)

# ----------------------------------------------------
# VIEW 1: FINANCIAL DASHBOARD (Phase 0)
# ----------------------------------------------------
if navigation == "Financial Dashboard":
    if df.empty:
        st.info("Upload data or use sample data to see the dashboard.")
    else:
        st.header("Financial Dashboard")

        # Calculate financial summary metrics
        summary = calculate_summary(df)
        st.session_state["financial_summary"] = summary

        # Display 4 KPI cards side by side
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        kpi1.metric("Total Revenue", f"${summary['total_revenue']:,.2f}")
        kpi2.metric("Total Expenses", f"${summary['total_expenses']:,.2f}")

        # Net Profit metric with built-in color delta
        net_profit_val = summary["net_profit"]
        kpi3.metric(
            "Net Profit",
            f"${net_profit_val:,.2f}",
            delta=f"${net_profit_val:,.2f}",
            delta_color="normal" if net_profit_val >= 0 else "inverse",
        )

        kpi4.metric("Profit Margin", f"{summary['profit_margin']:.1f}%")

        # 5th smaller metric / caption for Net Cash Flow
        st.caption(
            f"**Net Cash Flow**: ${summary['net_cash_flow']:,.2f} *(Simplified cash flow view: total revenue in minus total expenses out)*"
        )
        st.write("")

        # Calculate monthly summary for charts
        m_df = monthly_summary(df)
        st.session_state["monthly_summary"] = m_df

        if not m_df.empty:
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                # Grouped Bar Chart: Monthly Income vs Expenses
                fig_bar = go.Figure(data=[
                    go.Bar(name="Income", x=m_df["Month"], y=m_df["Income"], marker_color="#2ecc71"),
                    go.Bar(name="Expense", x=m_df["Month"], y=m_df["Expense"], marker_color="#e74c3c"),
                ])
                fig_bar.update_layout(
                    title="Monthly Income vs Expenses",
                    barmode="group",
                    xaxis_title="Month",
                    yaxis_title="Amount ($)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_chart2:
                # Line Chart: Monthly Net Profit Trend
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=m_df["Month"],
                    y=m_df["Net"],
                    mode="lines+markers",
                    name="Net Profit",
                    line=dict(color="#3498db", width=3),
                ))
                fig_line.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Profit Line")
                fig_line.update_layout(
                    title="Monthly Net Profit Trend",
                    xaxis_title="Month",
                    yaxis_title="Net Profit ($)",
                    showlegend=False,
                )
                st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # Expense Analysis Section
    st.header("Expense Analysis")

    # Check if DataFrame contains expense rows
    has_expenses = False
    if not df.empty and "Type" in df.columns:
        has_expenses = (df["Type"].astype(str).str.strip().str.title() == "Expense").any()

    if not has_expenses:
        st.info("No expense data available to analyze.")
    else:
        cat_df = expense_by_category(df)
        st.session_state["expense_breakdown"] = cat_df

        col_pie, col_tbl = st.columns([1, 1])

        with col_pie:
            # Plotly Donut Chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=cat_df["Category"],
                values=cat_df["Total"],
                hole=0.4,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Total: $%{value:,.2f}<br>Share: %{percent}<extra></extra>",
            )])
            fig_pie.update_layout(title="Expense Breakdown by Category")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_tbl:
            st.subheader("Category Breakdown Table")
            # Format table for clean UI display
            cat_display = cat_df.copy()
            cat_display["Total"] = cat_display["Total"].apply(lambda x: f"${x:,.2f}")
            cat_display["Percentage"] = cat_display["Percentage"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(cat_display, use_container_width=True)

        st.write("")

        # Month-over-Month Expense Flags
        monthly_cat_df = monthly_expense_by_category(df)
        flags = detect_expense_flags(monthly_cat_df)
        st.session_state["expense_flags"] = flags

        st.subheader("Month-over-Month Expense Flags")
        if flags:
            for flag in flags:
                prior_fmt = f"${flag['prior_month_amount']:,.2f}"
                curr_fmt = f"${flag['current_month_amount']:,.2f}"
                st.warning(
                    f"⚠️ **{flag['category']}** increased {flag['pct_change']}% month-over-month "
                    f"(from {prior_fmt} to {curr_fmt})."
                )
        else:
            st.success("No unusual expense increases detected this period.")

    st.markdown("---")

    # Financial Health Score Section
    st.header("Financial Health Score")

    summary = st.session_state.get("financial_summary")
    monthly_df = st.session_state.get("monthly_summary")
    flags = st.session_state.get("expense_flags", [])

    if not summary or (summary["total_revenue"] == 0 and summary["total_expenses"] == 0):
        st.info("Add transaction data to calculate a health score.")
    else:
        # Ensure monthly_df is available
        if monthly_df is None:
            monthly_df = monthly_summary(df)
            st.session_state["monthly_summary"] = monthly_df

        # Calculate overall health score
        health = calculate_health_score(summary, monthly_df, flags)
        st.session_state["health_score"] = health

        overall_score = health["overall_score"]
        rating = health["rating"]

        # Color indicator based on rating
        if rating in ["Excellent", "Good"]:
            rating_color = "#2ecc71"
            container_fn = st.success
        elif rating == "Fair":
            rating_color = "#f39c12"
            container_fn = st.warning
        else:
            rating_color = "#e74c3c"
            container_fn = st.error

        col_score_card, col_gauge = st.columns([1, 1])

        with col_score_card:
            st.write("")
            st.markdown(
                f"""
                <div style="border: 2px solid {rating_color}; border-radius: 10px; padding: 25px; text-align: center; background-color: rgba(255,255,255,0.05);">
                    <h3 style="margin-bottom: 0px; color: #888;">Overall Health Rating</h3>
                    <h1 style="font-size: 3.5rem; margin: 10px 0; color: {rating_color};">{overall_score} / 100</h1>
                    <h2 style="margin-top: 0px; color: {rating_color}; font-weight: bold;">{rating}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_gauge:
            # Plotly Gauge Indicator Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_score,
                title={"text": f"Financial Health Index ({rating})", "font": {"size": 18}},
                number={"suffix": " / 100", "font": {"size": 24, "color": rating_color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": rating_color},
                    "bgcolor": "white",
                    "borderwidth": 1,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 40], "color": "#fadbd8"},
                        {"range": [40, 60], "color": "#fdebd0"},
                        {"range": [60, 80], "color": "#d4efdf"},
                        {"range": [80, 100], "color": "#a9dfbf"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Expandable explainability section
        with st.expander("How is this score calculated?"):
            st.markdown("#### Component Scoring Breakdown")
            for comp_key, comp in health["components"].items():
                comp_name = comp_key.replace("_score", "").replace("_", " ").title()
                st.markdown(
                    f"• **{comp_name}**: **{comp['score']}/100** *(Weight: {comp['weight_pct']}%)* — {comp['explanation']}"
                )

    st.markdown("---")

    # AI Financial Recommendations Section
    st.header("AI Financial Recommendations")

    expense_breakdown = st.session_state.get("expense_breakdown")
    health_score = st.session_state.get("health_score")

    if not summary or (summary["total_revenue"] == 0 and summary["total_expenses"] == 0):
        st.info("Add transaction data to generate recommendations.")
    else:
        if st.button("Generate Recommendations"):
            with st.spinner("Analyzing your financials..."):
                ai_res, err = get_ai_recommendations(summary, expense_breakdown, flags, health_score)

                if err == "missing_key":
                    st.warning("AI API key not configured — showing rule-based recommendations instead.")
                    recs_text = get_fallback_recommendations(summary, expense_breakdown, flags, health_score)
                    st.session_state["recommendations_text"] = recs_text
                    st.session_state["recommendations_type"] = "fallback"
                elif err is not None and err.startswith("api_error"):
                    st.warning("AI service unavailable right now — showing rule-based recommendations instead.")
                    recs_text = get_fallback_recommendations(summary, expense_breakdown, flags, health_score)
                    st.session_state["recommendations_text"] = recs_text
                    st.session_state["recommendations_type"] = "fallback"
                else:
                    st.session_state["recommendations_text"] = ai_res
                    st.session_state["recommendations_type"] = "ai"

        # Render stored recommendations if available
        saved_recs = st.session_state.get("recommendations_text")
        rec_type = st.session_state.get("recommendations_type")

        if saved_recs:
            if rec_type == "ai":
                st.success("Generated by AI based on your data.")
            st.markdown(saved_recs)
            st.caption("Disclaimer: AI-generated recommendations are for informational purposes only and do not constitute formal financial advice.")

    st.markdown("---")

    # Transaction Data Section
    st.header("Transaction Data")

    if df.empty:
        st.warning("No transaction data found. Please check data/transactions.csv.")
    else:
        st.dataframe(df, use_container_width=True)
        st.write("")

        col1, col2, col3 = st.columns(3)
        total_count = len(df)
        col1.metric("Total Transactions", f"{total_count}")

        min_date_str = df["Date"].min().strftime("%b %d, %Y")
        max_date_str = df["Date"].max().strftime("%b %d, %Y")
        col2.metric("Date Range Covered", f"{min_date_str} - {max_date_str}")

        unique_cat_count = df["Category"].nunique()
        col3.metric("Unique Categories", f"{unique_cat_count}")

# ----------------------------------------------------
# VIEW 2: ANOMALY DETECTION (Phase 1)
# ----------------------------------------------------
elif navigation == "Anomaly Detection":
    st.header("Financial Anomaly Detection")
    st.caption("Identify unusual transaction variances, category spending spikes, and trend deviations requiring review.")

    if df.empty:
        st.info("Upload data or use sample data to analyze anomalies.")
    else:
        # Step 1: Detect Anomalies
        tx_anomalies = detect_transaction_anomalies(df, z_medium=2.5, z_high=3.0)
        cat_anomalies = detect_category_anomalies(df, threshold_pct=30.0)
        trend_anomalies = detect_trend_anomalies(df, threshold_pct=25.0)

        # Step 2: Summary KPI Cards
        total_anom = len(tx_anomalies) + len(cat_anomalies) + len(trend_anomalies)
        high_sev = (
            len([a for a in tx_anomalies if a["Severity"] == "High"]) +
            len([c for c in cat_anomalies if c["Severity"] == "High"]) +
            len([t for t in trend_anomalies if t["Severity"] == "High"])
        )
        med_sev = (
            len([a for a in tx_anomalies if a["Severity"] == "Medium"]) +
            len([c for c in cat_anomalies if c["Severity"] == "Medium"]) +
            len([t for t in trend_anomalies if t["Severity"] == "Medium"])
        )

        akpi1, akpi2, akpi3 = st.columns(3)
        akpi1.metric("Total Anomalies Identified", f"{total_anom}")
        akpi2.metric("High Severity Alerts", f"{high_sev}", delta="Requires Review" if high_sev > 0 else "Normal", delta_color="inverse")
        akpi3.metric("Medium Severity Alerts", f"{med_sev}")

        st.write("")

        # Step 3: Executive "What Needs Attention?" Section
        st.subheader("What Needs Attention?")

        if st.button("Generate AI Anomaly Analysis"):
            with st.spinner("Synthesizing anomaly insights..."):
                exp_text = get_anomaly_ai_explanation(tx_anomalies, cat_anomalies, trend_anomalies)
                st.session_state["anomaly_explanation_text"] = exp_text

        saved_exp = st.session_state.get("anomaly_explanation_text")
        if not saved_exp:
            saved_exp = get_anomaly_fallback_explanation(tx_anomalies, cat_anomalies, trend_anomalies)
            st.session_state["anomaly_explanation_text"] = saved_exp

        st.markdown(saved_exp)
        st.caption("Disclaimer: Anomaly alerts are statistical flags for administrative review and do not constitute proof of error or wrongdoing.")

        st.markdown("---")

        # Step 4: Transaction Anomalies Table
        st.subheader("Flagged Transaction Anomalies (Z-Score Variance)")
        if tx_anomalies:
            tx_df = pd.DataFrame(tx_anomalies)
            display_cols = ["Date", "Description", "Category", "Amount", "Expected/Average", "Z-score", "Severity", "Reason"]
            st.dataframe(tx_df[display_cols], use_container_width=True)
        else:
            st.success("No individual transaction anomalies detected above Z-score threshold (2.5).")

        st.write("")

        # Step 5: Visualizations
        st.subheader("Expense Amount Distribution & Anomaly Visualizer")
        exp_df = df[df["Type"].astype(str).str.strip().str.title() == "Expense"].copy()

        if not exp_df.empty:
            fig_scatter = go.Figure()

            # Normal vs Anomaly scatter trace
            flagged_set = set((a["Date"], a["Category"], a["Amount"]) for a in tx_anomalies)

            normal_xs, normal_ys, normal_txts = [], [], []
            anom_xs, anom_ys, anom_txts = [], [], []

            for _, row in exp_df.iterrows():
                d_str = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])
                c_str = str(row["Category"])
                a_val = float(row["Amount"])
                desc = str(row.get("Description", ""))

                if (d_str, c_str, a_val) in flagged_set:
                    anom_xs.append(c_str)
                    anom_ys.append(a_val)
                    anom_txts.append(f"{desc} (${a_val:,.2f})")
                else:
                    normal_xs.append(c_str)
                    normal_ys.append(a_val)
                    normal_txts.append(f"{desc} (${a_val:,.2f})")

            fig_scatter.add_trace(go.Scatter(
                x=normal_xs,
                y=normal_ys,
                mode="markers",
                name="Normal Transactions",
                marker=dict(size=9, color="#2ecc71", opacity=0.6),
                text=normal_txts,
                hovertemplate="<b>%{x}</b><br>Amount: $%{y:,.2f}<br>%{text}<extra></extra>",
            ))

            if anom_xs:
                fig_scatter.add_trace(go.Scatter(
                    x=anom_xs,
                    y=anom_ys,
                    mode="markers",
                    name="Flagged Anomaly",
                    marker=dict(size=14, color="#e74c3c", symbol="diamond", line=dict(width=2, color="#900c3f")),
                    text=anom_txts,
                    hovertemplate="<b>%{x} (FLAGGED)</b><br>Amount: $%{y:,.2f}<br>%{text}<extra></extra>",
                ))

            fig_scatter.update_layout(
                title="Category Expense Amounts vs Flagged Anomalies",
                xaxis_title="Category",
                yaxis_title="Amount ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")

        # Step 6: Category Spending Spikes & Macro Trend Alerts
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader("Category Spending Spikes")
            if cat_anomalies:
                cat_anom_df = pd.DataFrame(cat_anomalies)
                cat_cols = ["Category", "Current Amount", "Average Amount", "Percentage Difference", "Severity"]
                st.dataframe(cat_anom_df[cat_cols], use_container_width=True)
            else:
                st.info("No category spending spikes detected.")

        with col_c2:
            st.subheader("Macro Trend Alerts")
            if trend_anomalies:
                tr_df = pd.DataFrame(trend_anomalies)
                tr_cols = ["Period", "Metric", "Current Value", "Prior Value", "Percentage Change", "Severity"]
                st.dataframe(tr_df[tr_cols], use_container_width=True)
            else:
                st.info("No unusual macro trend shifts detected.")

# ----------------------------------------------------
# VIEW 3: RISK & FORECAST (Phase 2)
# ----------------------------------------------------
elif navigation == "Risk & Forecast":
    st.header("Risk & Financial Forecasting")
    st.caption("Short-term financial projections and risk indicator dashboard based on historical trends.")

    if df.empty:
        st.info("Upload data or use sample data to view forecasts and risk analysis.")
    else:
        # Step 1: Run Anomaly Detection (Phase 1 reuse) and Forecasting Engine (Phase 2)
        anomalies_tuple = (
            detect_transaction_anomalies(df, z_medium=2.5, z_high=3.0),
            detect_category_anomalies(df, threshold_pct=30.0),
            detect_trend_anomalies(df, threshold_pct=25.0),
        )
        forecast_res = generate_short_term_forecast(df, forecast_months=2, min_periods=2)
        risk_res = detect_financial_risks(df, forecast_res, anomalies=anomalies_tuple)

        # Step 2: Handle Insufficient Historical Data
        if not forecast_res.get("has_sufficient_data"):
            st.warning(f"⚠️ {forecast_res['message']}")
            st.info("Tip: Upload a transaction CSV with at least 2 consecutive monthly periods to unlock financial projections and risk indicators.")
        else:
            # Step 3: Forecast KPI Cards
            next_m = forecast_res["next_month"]
            st.subheader(f"Next Month Projections ({next_m.get('Month', 'Next Period')})")

            fkpi1, fkpi2, fkpi3 = st.columns(3)
            fkpi1.metric("Projected Revenue", f"${next_m['projected_revenue']:,.2f}")
            fkpi2.metric("Projected Expenses", f"${next_m['projected_expenses']:,.2f}")

            proj_profit = next_m["projected_net_profit"]
            fkpi3.metric(
                "Projected Net Profit",
                f"${proj_profit:,.2f}",
                delta=f"${proj_profit:,.2f}",
                delta_color="normal" if proj_profit >= 0 else "inverse",
            )

            st.write("")

            # Step 4: Financial Risk Overview Cards
            st.subheader("Financial Risk Overview")
            rc1, rc2, rc3, rc4 = st.columns(4)

            def get_sev_color(sev):
                if sev == "High":
                    return "#e74c3c"
                elif sev == "Medium":
                    return "#f39c12"
                return "#2ecc71"

            for col, risk_key, label in [
                (rc1, "revenue_risk", "Revenue Risk"),
                (rc2, "expense_risk", "Expense Risk"),
                (rc3, "profitability_risk", "Profitability Risk"),
                (rc4, "cash_flow_risk", "Cash Flow Risk"),
            ]:
                r_info = risk_res[risk_key]
                color = get_sev_color(r_info["severity"])
                col.markdown(
                    f"""
                    <div style="border: 2px solid {color}; border-radius: 8px; padding: 15px; text-align: center; background-color: rgba(255,255,255,0.03);">
                        <h4 style="margin: 0; color: #888;">{label}</h4>
                        <h2 style="margin: 5px 0; color: {color}; font-weight: bold;">{r_info['severity']}</h2>
                        <p style="font-size: 0.85rem; margin: 0; color: #bbb;">{r_info['reason']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            st.info(f"**Primary Financial Focus**: {risk_res['main_risk']}")
            st.write("")
            st.markdown("---")

            # Step 5: Executive "What Should I Watch?" Section
            st.subheader("What Should I Watch?")

            if st.button("Generate AI Risk & Forecast Analysis"):
                with st.spinner("Synthesizing risk & forecast insights..."):
                    exp_rf = get_risk_forecast_ai_explanation(forecast_res, risk_res)
                    st.session_state["risk_forecast_explanation_text"] = exp_rf

            saved_rf_exp = st.session_state.get("risk_forecast_explanation_text")
            if not saved_rf_exp:
                saved_rf_exp = get_risk_forecast_fallback_explanation(forecast_res, risk_res)
                st.session_state["risk_forecast_explanation_text"] = saved_rf_exp

            st.markdown(saved_rf_exp)
            st.caption("Disclaimer: Projections are estimates based on linear extrapolation of past trends and do not guarantee future financial results.")
            st.markdown("---")

            # Step 6: Visualizations (Historical Solid vs Projected Dashed)
            st.subheader("Historical vs Projected Financial Trends")

            hist_df = forecast_res["historical"]
            fc_df = forecast_res["forecast"]

            # Connection DataFrame connecting last historical point to first projected point
            last_hist_row = hist_df.iloc[-1]
            fc_conn_df = pd.concat([pd.DataFrame([last_hist_row]), fc_df], ignore_index=True)

            col_fc1, col_fc2 = st.columns(2)

            with col_fc1:
                # Revenue: Solid Historical vs Dashed Projected
                fig_rev = go.Figure()
                fig_rev.add_trace(go.Scatter(
                    x=hist_df["Month"],
                    y=hist_df["Income"],
                    mode="lines+markers",
                    name="Historical Revenue",
                    line=dict(color="#2ecc71", width=3),
                    marker=dict(size=8),
                ))
                fig_rev.add_trace(go.Scatter(
                    x=fc_conn_df["Month"],
                    y=fc_conn_df["Income"],
                    mode="lines+markers",
                    name="Projected Revenue",
                    line=dict(color="#2ecc71", width=3, dash="dash"),
                    marker=dict(size=10, symbol="circle-open"),
                ))
                fig_rev.update_layout(
                    title="Revenue: Historical vs Projected",
                    xaxis_title="Month",
                    yaxis_title="Amount ($)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_rev, use_container_width=True)

            with col_fc2:
                # Expenses: Solid Historical vs Dashed Projected
                fig_exp = go.Figure()
                fig_exp.add_trace(go.Scatter(
                    x=hist_df["Month"],
                    y=hist_df["Expense"],
                    mode="lines+markers",
                    name="Historical Expenses",
                    line=dict(color="#e74c3c", width=3),
                    marker=dict(size=8),
                ))
                fig_exp.add_trace(go.Scatter(
                    x=fc_conn_df["Month"],
                    y=fc_conn_df["Expense"],
                    mode="lines+markers",
                    name="Projected Expenses",
                    line=dict(color="#e74c3c", width=3, dash="dash"),
                    marker=dict(size=10, symbol="circle-open"),
                ))
                fig_exp.update_layout(
                    title="Expenses: Historical vs Projected",
                    xaxis_title="Month",
                    yaxis_title="Amount ($)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_exp, use_container_width=True)

            # Net Profit Trend: Solid Historical vs Dashed Projected
            fig_prof = go.Figure()
            fig_prof.add_trace(go.Scatter(
                x=hist_df["Month"],
                y=hist_df["Net"],
                mode="lines+markers",
                name="Historical Net Profit",
                line=dict(color="#3498db", width=3),
                marker=dict(size=8),
            ))
            fig_prof.add_trace(go.Scatter(
                x=fc_conn_df["Month"],
                y=fc_conn_df["Net"],
                mode="lines+markers",
                name="Projected Net Profit",
                line=dict(color="#3498db", width=3, dash="dash"),
                marker=dict(size=10, symbol="circle-open"),
            ))
            fig_prof.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Profit Baseline")
            fig_prof.update_layout(
                title="Net Profit: Historical vs Projected Trend",
                xaxis_title="Month",
                yaxis_title="Net Profit ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_prof, use_container_width=True)

# ----------------------------------------------------
# VIEW 4: WHAT-IF SIMULATOR (Phase 3)
# ----------------------------------------------------
elif navigation == "What-If Simulator":
    st.header("What-If Business Decision Simulator")
    st.caption("Simulate hypothetical financial outcomes based on custom revenue and expense change assumptions.")

    st.warning("⚠️ Disclaimer: This is a scenario simulation based on your assumptions, not a guaranteed financial prediction.")
    st.write("")

    if df.empty:
        st.info("Upload data or use sample data to run what-if simulations.")
    else:
        # Step 1: Calculate Current Baseline Summary
        baseline_summary = calculate_summary(df)
        st.session_state["baseline_summary"] = baseline_summary

        # Step 2: Baseline KPI Cards
        st.subheader("Current Financial Baseline Position")
        bkpi1, bkpi2, bkpi3, bkpi4 = st.columns(4)
        bkpi1.metric("Current Revenue", f"${baseline_summary['total_revenue']:,.2f}")
        bkpi2.metric("Current Expenses", f"${baseline_summary['total_expenses']:,.2f}")

        base_net = baseline_summary["net_profit"]
        bkpi3.metric(
            "Current Profit",
            f"${base_net:,.2f}",
            delta=f"${base_net:,.2f}",
            delta_color="normal" if base_net >= 0 else "inverse",
        )
        bkpi4.metric("Current Profit Margin", f"{baseline_summary['profit_margin']:.1f}%")

        st.caption(
            f"**Current Net Cash Flow**: ${baseline_summary['net_cash_flow']:,.2f} *(Simplified cash flow baseline: total revenue in minus total expenses out)*"
        )
        st.write("")
        st.markdown("---")

        # Step 3: Choose Scenario & Input Controls
        st.subheader("Choose Scenario Assumptions")

        presets = get_preset_scenarios()
        preset_names = list(presets.keys())

        selected_preset = st.selectbox(
            "Select Scenario Preset (or select Custom Scenario to set sliders):",
            options=preset_names,
            index=0,
        )

        preset_info = presets[selected_preset]
        st.caption(f"**Preset Description**: {preset_info['desc']}")
        st.write("")

        default_rev = preset_info["rev_change_pct"]
        default_exp = preset_info["exp_change_pct"]

        col_sl1, col_sl2 = st.columns(2)
        with col_sl1:
            rev_slider = st.slider(
                "Revenue Change (%)",
                min_value=-50.0,
                max_value=50.0,
                value=float(default_rev),
                step=1.0,
                help="Simulated percentage change in total revenue (-50% to +50%).",
            )
        with col_sl2:
            exp_slider = st.slider(
                "Expense Change (%)",
                min_value=-50.0,
                max_value=50.0,
                value=float(default_exp),
                step=1.0,
                help="Simulated percentage change in total expenses (-50% to +50%).",
            )

        # Step 4: Scenario Calculation Engine
        sim_res = calculate_scenario(baseline_summary, rev_change_pct=rev_slider, exp_change_pct=exp_slider)
        st.session_state["latest_scenario_res"] = sim_res

        base_data = sim_res["baseline"]
        scen_data = sim_res["scenario"]
        chg_data = sim_res["changes"]

        st.markdown("---")

        # Step 5: Scenario Comparison Table
        st.subheader("Scenario Results & Comparison")

        comp_df = pd.DataFrame([
            {
                "Financial Metric": "Total Revenue",
                "Current Baseline": f"${base_data['revenue']:,.2f}",
                "Scenario Projection": f"${scen_data['revenue']:,.2f}",
                "Absolute Change": f"${chg_data['rev_diff']:+,.2f}",
            },
            {
                "Financial Metric": "Total Expenses",
                "Current Baseline": f"${base_data['expenses']:,.2f}",
                "Scenario Projection": f"${scen_data['expenses']:,.2f}",
                "Absolute Change": f"${chg_data['exp_diff']:+,.2f}",
            },
            {
                "Financial Metric": "Net Profit",
                "Current Baseline": f"${base_data['profit']:,.2f}",
                "Scenario Projection": f"${scen_data['profit']:,.2f}",
                "Absolute Change": f"${chg_data['profit_diff']:+,.2f}",
            },
            {
                "Financial Metric": "Profit Margin",
                "Current Baseline": f"{base_data['margin']:.1f}%",
                "Scenario Projection": f"{scen_data['margin']:.1f}%",
                "Absolute Change": f"{chg_data['margin_diff_pp']:+.1f} percentage points",
            },
        ])

        st.dataframe(comp_df, use_container_width=True)
        st.write("")

        # Step 6: Visual Comparison Chart
        st.subheader("Current vs Scenario Comparison Visualizer")

        fig_sim = go.Figure(data=[
            go.Bar(
                name="Current Baseline",
                x=["Revenue", "Expenses", "Net Profit"],
                y=[base_data["revenue"], base_data["expenses"], base_data["profit"]],
                marker_color="#3498db",
                text=[f"${base_data['revenue']:,.0f}", f"${base_data['expenses']:,.0f}", f"${base_data['profit']:,.0f}"],
                textposition="auto",
            ),
            go.Bar(
                name="Scenario Projection",
                x=["Revenue", "Expenses", "Net Profit"],
                y=[scen_data["revenue"], scen_data["expenses"], scen_data["profit"]],
                marker_color="#2ecc71" if scen_data["profit"] >= base_data["profit"] else "#e74c3c",
                text=[f"${scen_data['revenue']:,.0f}", f"${scen_data['expenses']:,.0f}", f"${scen_data['profit']:,.0f}"],
                textposition="auto",
            ),
        ])

        fig_sim.update_layout(
            title="Current Baseline vs Simulated Scenario ($)",
            barmode="group",
            xaxis_title="Financial Metric",
            yaxis_title="Amount ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_sim, use_container_width=True)
        st.markdown("---")

        # Step 7: "What Changes?" & Business Impact Assessment
        col_wc1, col_wc2 = st.columns(2)

        interpretation = generate_business_interpretation(base_data, scen_data, chg_data)

        with col_wc1:
            st.subheader("What Changes?")
            rev_diff = chg_data["rev_diff"]
            exp_diff = chg_data["exp_diff"]
            profit_diff = chg_data["profit_diff"]
            margin_diff = chg_data["margin_diff_pp"]

            rev_msg = f"Revenue would increase by +${rev_diff:,.2f}" if rev_diff > 0 else (f"Revenue would decrease by -${abs(rev_diff):,.2f}" if rev_diff < 0 else "Revenue would remain unchanged")
            exp_msg = f"Expenses would increase by +${exp_diff:,.2f}" if exp_diff > 0 else (f"Expenses would decrease by -${abs(exp_diff):,.2f}" if exp_diff < 0 else "Expenses would remain unchanged")
            profit_msg = f"Estimated profit would improve by +${profit_diff:,.2f}" if profit_diff > 0 else (f"Estimated profit would decline by -${abs(profit_diff):,.2f}" if profit_diff < 0 else "Estimated profit would remain unchanged")
            margin_msg = f"Profit margin would improve by +{margin_diff:.1f} percentage points" if margin_diff > 0 else (f"Profit margin would decline by {margin_diff:.1f} percentage points" if margin_diff < 0 else "Profit margin would remain unchanged")

            st.markdown(f"• **{rev_msg}.**")
            st.markdown(f"• **{exp_msg}.**")
            st.markdown(f"• **{profit_msg}.**")
            st.markdown(f"• **{margin_msg}.**")

        with col_wc2:
            st.subheader("Business Impact Assessment")
            if scen_data["profit"] < 0:
                st.error(f"⚠️ {interpretation}")
            elif profit_diff > 0:
                st.success(f"✅ {interpretation}")
            else:
                st.warning(f"⚠️ {interpretation}")

        st.write("")

        # Step 8: Optional AI Commentary
        if st.button("Generate AI Scenario Commentary"):
            with st.spinner("Synthesizing scenario commentary..."):
                exp_sim = get_simulator_ai_explanation(base_data, scen_data, chg_data, interpretation)
                st.session_state["simulator_explanation_text"] = exp_sim

        saved_sim_exp = st.session_state.get("simulator_explanation_text")
        if not saved_sim_exp:
            saved_sim_exp = get_simulator_fallback_explanation(base_data, scen_data, chg_data, interpretation)
            st.session_state["simulator_explanation_text"] = saved_sim_exp

        st.markdown(saved_sim_exp)
        st.caption("This is a scenario simulation based on your assumptions, not a guaranteed financial prediction.")

# ----------------------------------------------------
# VIEW 5: AI FINANCIAL COPILOT (Phase 4)
# ----------------------------------------------------
elif navigation == "AI Financial Copilot":
    st.header("AI Financial Copilot")
    st.caption("Intelligent decision-support copilot interpreting overall financial results, anomalies, risks, forecasts, and What-If scenarios.")

    st.caption("Disclaimer: AI-generated explanations are based on the available financial data and should be reviewed before making business decisions.")
    st.write("")

    if df.empty:
        st.info("Upload data or use sample data to interact with the AI Financial Copilot.")
    else:
        # Step 1: Assemble Context from Phase 0, 1, 2, and 3
        summary = calculate_summary(df)
        m_df = monthly_summary(df)
        flags = detect_expense_flags(monthly_expense_by_category(df))
        health = calculate_health_score(summary, m_df, flags)
        anomalies_tuple = (
            detect_transaction_anomalies(df, z_medium=2.5, z_high=3.0),
            detect_category_anomalies(df, threshold_pct=30.0),
            detect_trend_anomalies(df, threshold_pct=25.0),
        )
        forecast_res = generate_short_term_forecast(df, forecast_months=2, min_periods=2)
        risk_res = detect_financial_risks(df, forecast_res, anomalies=anomalies_tuple)
        scenario_res = st.session_state.get("latest_scenario_res")

        copilot_context = build_financial_context(
            df=df,
            summary=summary,
            health=health,
            anomalies=anomalies_tuple,
            forecast=forecast_res,
            risk=risk_res,
            scenario=scenario_res,
        )
        st.session_state["copilot_context"] = copilot_context

        # Step 2: AI Financial Summary & Recommendations Section
        st.subheader("AI Financial Summary & Recommendations")

        if st.button("Generate Financial Summary & Recommendations"):
            with st.spinner("Analyzing overall financial results..."):
                copilot_summary_output = generate_copilot_financial_summary(copilot_context)
                st.session_state["copilot_summary_output"] = copilot_summary_output

        saved_copilot_sum = st.session_state.get("copilot_summary_output")
        if not saved_copilot_sum:
            saved_copilot_sum = generate_rule_based_copilot_summary(copilot_context)
            st.session_state["copilot_summary_output"] = saved_copilot_sum

        st.markdown(saved_copilot_sum)
        st.markdown("---")

        # Step 3: Ask Your Financial Copilot Section
        st.subheader("Ask Your Financial Copilot")
        st.caption("Ask specific questions about revenue, expenses, profit, margins, cash flow, anomalies, risks, forecasts, or What-If scenarios.")

        st.markdown("**Suggested Questions:**")
        q_col1, q_col2, q_col3 = st.columns(3)

        quick_q = None
        if q_col1.button("What should I look at first?"):
            quick_q = "What should I look at first?"
        if q_col2.button("Why is my profit decreasing?"):
            quick_q = "Why is my profit decreasing?"
        if q_col3.button("Which expense area needs attention?"):
            quick_q = "Which expense area needs attention?"

        q_col4, q_col5 = st.columns(2)
        if q_col4.button("What is causing my financial risk?"):
            quick_q = "What is causing my financial risk?"
        if q_col5.button("How can I improve profitability?"):
            quick_q = "How can I improve profitability?"

        user_q_input = st.text_input("Or enter a custom question:", value=quick_q if quick_q else "", placeholder="e.g. What should I look at first?")

        if st.button("Ask Copilot") or quick_q:
            target_q = quick_q if quick_q else user_q_input
            if target_q and target_q.strip():
                with st.spinner(f"Copilot analyzing question: '{target_q}'..."):
                    ans_output = answer_copilot_question(copilot_context, target_q)
                    st.session_state["latest_copilot_q"] = target_q
                    st.session_state["latest_copilot_ans"] = ans_output

        saved_q = st.session_state.get("latest_copilot_q")
        saved_ans = st.session_state.get("latest_copilot_ans")

        if saved_ans:
            st.markdown(f"**Question**: *\"{saved_q}\"*")
            st.info(f"**Copilot Answer**:\n\n{saved_ans}")

        st.markdown("---")

        # Step 4: Context Data Transparency Section
        with st.expander("View Summarized Financial Context Sent to Copilot"):
            st.caption("To protect data privacy, only compact summary metrics (never raw transaction CSV logs) are sent to the AI Copilot.")
            st.code(copilot_context.get("summary_text", "No context available"), language="markdown")
