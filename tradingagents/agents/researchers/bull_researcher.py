

def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        quant_report = state.get("quant_report", "")
        verified_data = state.get("verified_data", "")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}\n\n{quant_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

LEVERAGE / D/E INTERPRETATION RULE (HARD PROHIBITION):
If the fundamentals report contains a DEBT DECOMPOSITION showing a NET CASH position
(Total Cash > Total Debt), the D/E ratio MUST NOT appear as a risk factor ANYWHERE.
FORBIDDEN PHRASES when net cash > 0: "high D/E," "high leverage," "alarming,"
"precarious," "financial distress," "lack of financial flexibility/resilience."
The company HAS MORE CASH THAN DEBT — cite the net cash position as proof of
financial resilience. Counter any bear D/E argument by stating: "D/E of X is a
mathematical artifact of thin equity from accumulated losses; the company has net
cash of $Y, meaning it could repay ALL debt and still have cash remaining."

EARNINGS GROWTH INTEGRITY RULE:
If you cite earnings growth >50% YoY as a positive signal, you MUST first check:
(a) Did revenue grow proportionally? If revenue declined or was flat, earnings growth
    is cost-driven, not revenue-driven — state this explicitly.
(b) Were prior-year earnings depressed by one-time charges (restructuring, impairments)?
    If so, the YoY comparison overstates sustainable improvement.
(c) FORBIDDEN: citing large earnings growth as "turnaround evidence" or "momentum"
    without decomposing into sustainable vs one-time components.

REVENUE METRIC CONSISTENCY RULE:
You MUST use the same revenue growth metric as the verified data. If the verified data
shows Revenue CAGR (1yr) and quarterly revenue trend, use those — do NOT cherry-pick a
different time window that flatters your bull case. If annual revenue declined 5% but
you cite -0.9% from a different window, you MUST reconcile the discrepancy.

NEWS/SENTIMENT DATA INTEGRITY RULE:
If the sentiment report says "INSUFFICIENT DATA" or "DATA PIPELINE FAILURE," you MUST NOT
characterise sentiment as "stable," "neutral," or "not negative." Absent data is UNKNOWN,
not positive. Do NOT cite lack of negative news as a bull argument when the data pipeline
failed to retrieve any news at all.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

BALANCE SHEET GROUND TRUTH (HARD — NO EXCEPTIONS):
The verified data below contains independently verified balance sheet figures
(Total Debt, Total Cash, Net Debt/Cash, D/E Ratio, Equity Ratio).
You MUST use ONLY these verified figures for any claim about the company's
financial position. Do NOT invent, reinterpret, or contradict the verified data.
If verified data shows Net Debt of $X, you CANNOT claim the company has a
'net cash position.' Doing so is a FACTUAL FABRICATION.

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}

=== VERIFIED GROUND-TRUTH DATA ===
{verified_data}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. You must also address reflections and learn from lessons and mistakes you made in the past.
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
