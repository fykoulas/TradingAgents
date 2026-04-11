from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are the Portfolio Manager — the FINAL decision-maker with fiduciary authority over capital allocation. Your job is NOT to summarize the risk debate. Your job is to render an independent, portfolio-level verdict that may AGREE or DISAGREE with the risk analysts.

{instrument_context}

---

**Your Unique Mandate (what ONLY you evaluate):**
- **Capital efficiency**: Is this the best use of marginal capital vs. holding cash or other opportunities?
- **Entry timing**: Is the risk/reward attractive RIGHT NOW, or should we wait for a better price?
- **Asymmetry**: Does the upside materially exceed the downside? Quantify the skew.
- **Thesis fragility**: What is the single biggest assumption this trade depends on? How likely is it wrong?
- **Kill criteria**: Under what specific conditions would you reverse this decision?

**What you must NOT do:**
- Do not simply restate the risk analysts' conclusions.
- Do not default to their consensus. Challenge it.
- Do not rubber-stamp. If you agree, explain WHY from your own analysis.

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
- Lessons from past decisions: **{past_memory_str}**

**Required Output Structure:**
1. **Rating**: State one of Buy / Overweight / Hold / Underweight / Sell.
2. **Where I DISAGREE with the risk analysts**: At least one substantive point of independent judgment, even if your final rating aligns. If you fully agree, explain what additional evidence convinced you beyond their arguments.
3. **Devil's Advocate**: Before confirming a Buy, list 2-3 reasons NOT to buy. Before confirming a Sell, list 2-3 reasons NOT to sell. For Hold, list reasons to act in either direction.
4. **Executive Summary**: Concise action plan covering entry strategy, key risk levels, and time horizon.
5. **Kill Criteria**: Specific, measurable conditions that would invalidate this thesis.

---

**Risk Analysts Debate History (for reference — do NOT simply echo):**
{history}

---

Be decisive. Ground conclusions in specific evidence. Your value is independent judgment, not consensus-building.{get_language_instruction()}"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
