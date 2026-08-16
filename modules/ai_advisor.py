"""
ai_advisor.py
Interfaces with Google Gemini API using official google-genai SDK to generate AI financial recommendations.
"""

import os
from dotenv import load_dotenv


def build_prompt(summary, expense_breakdown, expense_flags, health_score):
    """
    Constructs a detailed structured prompt incorporating financial metrics, expense breakdown,
    flags, and health score to send to the AI model.
    """
    prompt = f"""
You are an expert financial consultant for small businesses. Analyze the following financial dataset and health evaluation, then provide clear, actionable financial recommendations.

### 1. FINANCIAL SUMMARY:
- Total Revenue: ${summary.get('total_revenue', 0.0):,.2f}
- Total Expenses: ${summary.get('total_expenses', 0.0):,.2f}
- Net Profit: ${summary.get('net_profit', 0.0):,.2f}
- Profit Margin: {summary.get('profit_margin', 0.0):.1f}%
- Net Cash Flow: ${summary.get('net_cash_flow', 0.0):,.2f}

### 2. FINANCIAL HEALTH EVALUATION:
- Overall Health Score: {health_score.get('overall_score', 0)} / 100
- Health Rating: {health_score.get('rating', 'Unknown')}
"""
    if "components" in health_score:
        prompt += "- Component Details:\n"
        for k, v in health_score["components"].items():
            prompt += f"  * {k}: {v.get('score', 0)}/100 (Weight {v.get('weight_pct', 0)}%) - {v.get('explanation', '')}\n"

    prompt += "\n### 3. EXPENSE BREAKDOWN:\n"
    if expense_breakdown is not None and not expense_breakdown.empty:
        for _, row in expense_breakdown.iterrows():
            prompt += f"- {row['Category']}: ${row['Total']:,.2f} ({row['Percentage']:.1f}% of expenses)\n"
    else:
        prompt += "No category breakdown available.\n"

    prompt += "\n### 4. EXPENSE ANOMALIES & FLAGS:\n"
    if expense_flags:
        for flag in expense_flags:
            prompt += f"- WARNING: {flag['category']} increased by {flag['pct_change']}% MoM (from ${flag['prior_month_amount']:,.2f} to ${flag['current_month_amount']:,.2f})\n"
    else:
        prompt += "No unusual expense spikes detected.\n"

    prompt += """
### INSTRUCTIONS FOR YOUR RECOMMENDATIONS:
Provide a structured, executive-level recommendation report with:
1. Executive Summary (2-3 sentences)
2. Top 3 Strategic Recommendations (bullet points with concrete actions)
3. Risk Mitigation & Cost Optimization Steps
4. Growth Opportunities
Keep your tone professional, encouraging, and clear.
"""
    return prompt


def get_ai_recommendations(summary, expense_breakdown, expense_flags, health_score):
    """
    Loads GEMINI_API_KEY and GEMINI_MODEL from environment using python-dotenv
    (load_dotenv() + os.getenv). GEMINI_MODEL should default to "gemini-3.5-flash-lite"
    if not set in .env.

    If GEMINI_API_KEY is missing, empty, or still equals the placeholder "PASTE_KEY_HERE",
    return (None, "missing_key") immediately without attempting an API call.

    Otherwise:
      - Build the prompt using build_prompt()
      - Create a client using the google-genai SDK: `from google import genai`, then
        `client = genai.Client(api_key=GEMINI_API_KEY)`
      - Call `client.models.generate_content(model=GEMINI_MODEL, contents=prompt)`
      - On success, extract the plain text from the response (the SDK's response object
        exposes a `.text` attribute) and return (response_text: str, None)
      - On any exception (network error, auth error, rate limit, invalid model name, etc.),
        catch it and return (None, f"api_error: {short description of the error}")
    """
    # Step 1: Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # Step 2: Validate API key existence and check against default placeholder
    if not api_key or api_key.strip() == "" or api_key.strip() == "PASTE_KEY_HERE":
        return (None, "missing_key")

    # Step 3: Invoke Google Gemini API using google-genai SDK safely inside try/except block
    try:
        from google import genai

        prompt = build_prompt(summary, expense_breakdown, expense_flags, health_score)

        # Initialize Google Gen AI client
        client = genai.Client(api_key=api_key.strip())

        # Generate content using configured Gemini model
        response = client.models.generate_content(
            model=model_name.strip(),
            contents=prompt,
        )

        # Extract text attribute from SDK response object
        if response and hasattr(response, "text") and response.text:
            return (response.text.strip(), None)
        else:
            return (None, "api_error: Empty response received from Gemini API.")

    except Exception as e:
        # Step 4: Catch all exceptions safely without crashing app
        return (None, f"api_error: {str(e)}")


def get_fallback_recommendations(summary, expense_breakdown, expense_flags, health_score):
    """
    Rule-based recommendation fallback function when AI API key is missing or API fails.
    Analyzes summary, expense flags, and health score to produce deterministic recommendations.
    """
    recs = []

    # 1. Net Profit & Margin Check
    margin = summary.get("profit_margin", 0.0)
    profit = summary.get("net_profit", 0.0)
    if margin < 10.0:
        recs.append(
            f"**Improve Operating Margin**: Your profit margin is currently {margin:.1f}%. Focus on reducing non-essential expenses and renegotiating vendor rates to raise margin above 15%."
        )
    else:
        recs.append(
            f"**Maintain Healthy Margin**: Your profit margin of {margin:.1f}% is solid. Reinvest excess profit of ${profit:,.2f} into high-ROI sales channels."
        )

    # 2. Expense Flag Anomaly Check
    if expense_flags:
        for flag in expense_flags:
            recs.append(
                f"**Audit Spiking Category ({flag['category']})**: {flag['category']} increased by {flag['pct_change']}% MoM (${flag['prior_month_amount']:,.2f} -> ${flag['current_month_amount']:,.2f}). Audit recent purchases in this category for cost leaks."
            )
    else:
        recs.append(
            "**Expense Control**: No unusual month-over-month category spikes detected. Continue monitoring category budgets monthly."
        )

    # 3. Overall Health Score Advice
    overall = health_score.get("overall_score", 50)
    rating = health_score.get("rating", "Fair")
    if overall < 60:
        recs.append(
            f"**Health Score Action ({overall}/100 - {rating})**: Build cash reserves immediately to cover at least 3 months of operating expenses."
        )
    else:
        recs.append(
            f"**Health Score Action ({overall}/100 - {rating})**: Strong financial foundation. Consider strategic expansion or automated inventory management."
        )

    fallback_text = "### Executive Recommendations (Rule-Based Fallback)\n\n"
    for i, rec in enumerate(recs, 1):
        fallback_text += f"{i}. {rec}\n\n"

    return fallback_text
