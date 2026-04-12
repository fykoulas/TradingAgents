

def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains.

LEVERAGE / D/E INTERPRETATION RULE (TWO EXCLUSIVE PATHS — CHECK FUNDAMENTALS DATA FIRST):
Step 1: Find Total Debt and Total Cash in the Company Fundamentals Report.
Step 2: Pick the CORRECT path below. There is NO third option.

PATH A — NET DEBT (Total Debt > Total Cash):
  The company carries net debt. You MAY cite leverage risk, debt burden,
  interest coverage concerns. These are legitimate conservative arguments.
  Use Net Debt/FCF as the primary leverage metric (not D/E when equity is
  compressed by buybacks). Net Debt/FCF ≥ 5x is genuinely elevated.
  Do NOT call D/E "uninformative" when the company has real net debt.

PATH B — NET CASH (Total Cash > Total Debt):
  The company has more cash than debt. D/E MUST NOT appear as a risk or concern.
  FORBIDDEN phrases under Path B:
  "high D/E," "high leverage," "precarious," "financial distress,"
  "lack of financial flexibility," "concerns about leverage," "financial instability."
  The ONLY permitted concern: "thin equity from accumulated operating losses" —
  a profitability problem, NOT a leverage problem.

VIOLATION OF PATH SELECTION = CRITICAL ANALYTICAL ERROR = ANALYSIS INVALID.

EARNINGS & REVENUE INTEGRITY RULES:
1. If the bull/aggressive side cites large earnings growth (>50% YoY), challenge whether
   it is sustainable — check revenue trajectory and prior-year one-time charges.
2. Use verified annual revenue figures as the authoritative growth metric.
   If verified annual revenue declined 5%, do NOT accept -0.9% as the growth characterisation.
3. If news/sentiment data is INSUFFICIENT or shows DATA PIPELINE FAILURE, flag this:
   absent data is a risk factor (unknown), not a neutral signal.

Here is the trader's decision:

{trader_decision}

Your task is to actively counter the arguments of the Aggressive and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} Here is the last response from the aggressive analyst: {current_aggressive_response} Here is the last response from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Focus exclusively on the risk/reward profile of THIS specific trade — do not make portfolio-level allocation decisions (that is the Portfolio Manager's role). Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting."""

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
