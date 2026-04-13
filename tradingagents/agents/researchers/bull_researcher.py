

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

LEVERAGE / D/E INTERPRETATION RULE (TWO EXCLUSIVE PATHS — CHECK VERIFIED DATA FIRST):
Step 1: Read the golden variables block at the bottom. Find "Net Debt" or "Net Cash."
Step 2: Pick the CORRECT path below. There is NO third option.

PATH A — NET DEBT (Total Debt > Total Cash):
  The company carries net debt. You MUST NOT claim it has a "net cash position,"
  "more cash than debt," or that it "could repay all debt." These are FABRICATIONS.
  If D/E is high, acknowledge leverage honestly. Do not minimize it.
  USE Net Debt/FCF from golden variables as the PRIMARY leverage metric — D/E is
  unreliable when equity is compressed by buybacks. You may argue the debt is
  manageable IF Net Debt/FCF < 4x (comfortable) or 4-6x (elevated but serviceable).
  If Net Debt/FCF ≥ 6x, do NOT call leverage "manageable" without citing specific
  interest coverage or debt maturity data.
  FORBIDDEN phrases under Path A:
  "net cash," "cash exceeds debt," "repay all debt," "debt-free,"
  "more cash than debt," "cash remaining after repaying debt,"
  "D/E is uninformative" (when net debt is real, D/E is misleading but leverage is NOT).

PATH B — NET CASH (Total Cash > Total Debt):
  The company has more cash than debt. D/E may be mathematically inflated by thin
  equity from accumulated losses — this is a profitability concern, not a solvency
  risk. You may cite the net cash position as financial resilience.
  FORBIDDEN phrases under Path B:
  "high leverage," "alarming D/E," "financial distress," "precarious."

VIOLATION OF PATH SELECTION = FACTUAL FABRICATION = ENTIRE ANALYSIS INVALID.

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

PRODUCT NAME & COMPETITIVE ATTRIBUTION (MANDATORY):
Do NOT cite specific product names, drug names, or brand names unless they
appear in the analyst reports or verified data provided to you. If a report
mentions a product, verify which company ACTUALLY manufactures it before
attributing competitive dynamics. Generic competition for a COMPETITOR's product
is not a risk to THIS company. If you cannot verify ownership, state it.

CAPEX-ARTIFACT LEVERAGE:
If verified data flags '⚠ CAPEX-ARTIFACT LEVERAGE', use Net Debt/OCF (not
Net Debt/FCF) as the leverage metric. The FCF-based ratio is distorted by
peak capex and does NOT reflect debt-serviceability.

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
