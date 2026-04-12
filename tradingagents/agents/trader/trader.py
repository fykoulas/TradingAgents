import functools

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        verified_data = state.get("verified_data", "")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. {instrument_context} This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\n{verified_data}\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation.

LEVERAGE / D/E INTERPRETATION RULE (TWO EXCLUSIVE PATHS — CHECK FUNDAMENTALS DATA FIRST):
Step 1: Find Total Debt and Total Cash in the fundamentals report.
Step 2: Pick the CORRECT path. There is NO third option.

PATH A — NET DEBT (Total Debt > Total Cash):
  Leverage is real. Use Net Debt/FCF as the primary leverage metric.
  If Net Debt/FCF ≥ 5x, factor elevated leverage into risk assessment.
  FORBIDDEN: claiming "net cash," "could repay all debt," "more cash than debt,"
  or calling D/E "uninformative" when net debt is substantial.

PATH B — NET CASH (Total Cash > Total Debt):
  D/E is a mathematical artifact. NOT a risk factor.
  FORBIDDEN: "high D/E," "high leverage," "financial instability," "concerns about leverage."

VIOLATION = FACTUAL FABRICATION = DECISION INVALID.

EARNINGS & REVENUE INTEGRITY RULES:
1. If earnings growth >50% YoY: verify whether revenue grew proportionally. If not,
   earnings growth is cost-driven/one-time and should NOT anchor your thesis.
2. Use the verified annual revenue data as the authoritative growth figure.
   Do NOT use a flattering quarterly comparison when the annual trend is worse.
3. If news/sentiment reports show INSUFFICIENT DATA or DATA PIPELINE FAILURE,
   do NOT treat sentiment as neutral or supportive in your trading plan.

Apply lessons from past decisions to strengthen your analysis. Here are reflections from similar situations you traded in and the lessons learned: {past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
