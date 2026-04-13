

def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies.

LEVERAGE / D/E INTERPRETATION RULE (TWO EXCLUSIVE PATHS — CHECK FUNDAMENTALS DATA FIRST):
Step 1: Find Total Debt and Total Cash in the Company Fundamentals Report.
Step 2: Pick the CORRECT path below. There is NO third option.

PATH A — NET DEBT (Total Debt > Total Cash):
  The company carries net debt. Leverage is a legitimate risk factor.
  Weigh it fairly using Net Debt/FCF (not D/E when equity is compressed).
  If either side misstates the debt position, flag it as a factual error.
  If either side calls D/E "uninformative" when Net Debt/FCF ≥ 3x, flag it —
  the debt IS real, only the D/E RATIO is uninformative.

PATH B — NET CASH (Total Cash > Total Debt):
  D/E MUST NOT appear as a risk or concern. The ONLY balanced framing:
  "Book D/E is elevated by thin equity from accumulated losses, but the company
  has net cash of $X — no near-term debt distress risk."
  If either side cited D/E as a leverage risk for a net-cash company, flag it
  as an analytical error.

VIOLATION OF PATH SELECTION = FACTUAL FABRICATION = FLAG IT.

PRODUCT & COMPETITIVE VERIFICATION:
Do NOT cite product/drug names unless they appear in the source reports. Do NOT
attribute competitive threats from a COMPETITOR's product to THIS company. If verified
data flags '⚠ CAPEX-ARTIFACT LEVERAGE', use Net Debt/OCF — not Net Debt/FCF.

EARNINGS & REVENUE INTEGRITY RULES:
1. If either side cites earnings growth >50% YoY without decomposing it (revenue-driven
   vs cost-driven vs one-time), flag this as incomplete analysis.
2. Ensure revenue growth figures used in the debate match the verified annual data.
   Reconcile any discrepancy between CAGR, YoY, and quarterly trends.
3. If news/sentiment data is INSUFFICIENT or shows DATA PIPELINE FAILURE, ensure neither
   side treats this as benign. Absent data is an unresolved risk, not neutral.

Here is the trader's decision:

{trader_decision}

Your task is to challenge both the Aggressive and Conservative Analysts, pointing out where each perspective may be overly optimistic or overly cautious. Use insights from the following data sources to support a moderate, sustainable strategy to adjust the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} Here is the last response from the aggressive analyst: {current_aggressive_response} Here is the last response from the conservative analyst: {current_conservative_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Focus exclusively on the risk/reward profile of THIS specific trade — do not make portfolio-level allocation decisions (that is the Portfolio Manager's role). Engage actively by analyzing both sides critically, addressing weaknesses in the aggressive and conservative arguments to advocate for a more balanced approach. Challenge each of their points to illustrate why a moderate risk strategy might offer the best of both worlds, providing growth potential while safeguarding against extreme volatility. Focus on debating rather than simply presenting data, aiming to show that a balanced view can lead to the most reliable outcomes. Output conversationally as if you are speaking without any special formatting."""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
