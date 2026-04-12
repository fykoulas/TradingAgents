

def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Aggressive Risk Analyst, your role is to actively champion high-reward, high-risk opportunities, emphasizing bold strategies and competitive advantages. When evaluating the trader's decision or plan, focus intently on the potential upside, growth potential, and innovative benefits—even when these come with elevated risk. Use the provided market data and sentiment analysis to strengthen your arguments and challenge the opposing views.

LEVERAGE / D/E INTERPRETATION RULE (TWO EXCLUSIVE PATHS — CHECK FUNDAMENTALS DATA FIRST):
Step 1: Find Total Debt and Total Cash in the Company Fundamentals Report.
Step 2: Pick the CORRECT path below. There is NO third option.

PATH A — NET DEBT (Total Debt > Total Cash):
  The company carries net debt. You MUST NOT claim it has a "net cash position."
  If D/E is high, acknowledge leverage honestly — then argue the debt is manageable
  using Net Debt/FCF (the correct metric when equity is compressed by buybacks).
  If Net Debt/FCF < 4x, argue comfortably serviceable. If 4-6x, acknowledge elevated.
  FORBIDDEN phrases under Path A:
  "net cash," "cash exceeds debt," "repay all debt," "debt-free,"
  "D/E is uninformative" (when net debt is real, leverage IS real).

PATH B — NET CASH (Total Cash > Total Debt):
  The company has more cash than debt. D/E is a mathematical artifact of thin equity.
  Counter conservative D/E concerns by citing the net cash position.
  FORBIDDEN phrases under Path B:
  "high leverage," "alarming D/E," "financial distress," "precarious."

VIOLATION OF PATH SELECTION = FACTUAL FABRICATION = ENTIRE ANALYSIS INVALID.

EARNINGS & REVENUE INTEGRITY RULES:
1. If you cite earnings growth >50% YoY, you MUST check whether revenue grew proportionally.
   Cost-driven earnings growth without revenue growth is NOT sustainable momentum.
2. Use the verified annual revenue data, not a cherry-picked quarterly comparison.
   If verified annual revenue declined 5% but a different window shows -0.9%, use the annual figure.
3. If the sentiment/news data shows INSUFFICIENT DATA or DATA PIPELINE FAILURE,
   do NOT treat sentiment as supportive. Absent data is unknown, not positive.

Specifically, respond directly to each point made by the conservative and neutral analysts, countering with data-driven rebuttals and persuasive reasoning. Highlight where their caution might miss critical opportunities or where their assumptions may be overly conservative. Here is the trader's decision:

{trader_decision}

Your task is to create a compelling case for the trader's decision by questioning and critiquing the conservative and neutral stances to demonstrate why your high-reward perspective offers the best path forward. Incorporate insights from the following sources into your arguments:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Here is the current conversation history: {history} Here are the last arguments from the conservative analyst: {current_conservative_response} Here are the last arguments from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Focus exclusively on the risk/reward profile of THIS specific trade — do not make portfolio-level allocation decisions (that is the Portfolio Manager's role). Engage actively by addressing any specific concerns raised, refuting the weaknesses in their logic, and asserting the benefits of risk-taking to outpace market norms. Maintain a focus on debating and persuading, not just presenting data. Challenge each counterpoint to underscore why a high-risk approach is optimal. Output conversationally as if you are speaking without any special formatting."""

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
