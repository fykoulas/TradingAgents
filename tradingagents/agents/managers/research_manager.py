
from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        verified_data = state.get("verified_data", "")

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Research Manager, your role is to critically evaluate the bull/bear debate and deliver a definitive investment recommendation.

{instrument_context}

{verified_data}

---

**SECTOR-AWARE RATIO INTERPRETATION:**
If the company is in Insurance, Banking, or REITs, standard financial ratios (D/E, Current Ratio, P/E) are structurally different from industrial companies. Do NOT treat a high D/E at an insurer or bank as a red flag without acknowledging that policyholder reserves (insurance) or deposits (banking) inflate reported liabilities. If an analyst report calls a financial-sector D/E "conservative" or "low" without sector context, flag it as an analytical error in your evaluation.

---

**QUANTITATIVE REALITY CHECK (evaluate BEFORE reading the debate):**
Review the verified market data above. Before proceeding, answer:
- Is the stock above or below the 200-day SMA? By how much?
- What is the 6-month return?
- Is the RSI indicating oversold (<30), neutral (30-70), or overbought (>70)?
A BUY recommendation on a stock that is >10% below the 200-day SMA with no confirmed reversal signal requires EXCEPTIONAL justification beyond narrative conviction.

---

**MANDATORY — Dissent Section (do this FIRST, before your recommendation):**
Before stating your recommendation, you MUST complete this structured dissent analysis:

1. **Strongest 3 arguments AGAINST your recommendation**: List the most compelling reasons from the opposing side of the debate. Do not dismiss them — steel-man them.
2. **What would change your mind**: State 1-2 specific, measurable conditions that would flip your recommendation (e.g., "If revenue growth drops below 10% next quarter" or "If the stock breaks below $X support").
3. **Confidence qualifier**: Rate your confidence as HIGH (>80% the debate evidence clearly favours one side), MEDIUM (60-80% evidence is mixed but leans), or LOW (<60% evidence is genuinely ambiguous). If LOW, your recommendation MUST be Hold.

---

**Then provide your recommendation:**

- **Your Recommendation**: Buy, Sell, or Hold — a decisive stance grounded in the debate's strongest arguments. Avoid defaulting to Hold simply because both sides have valid points.
- **Rationale**: Why these arguments lead to your conclusion, explicitly addressing why the dissenting evidence is insufficient to change your mind.
- **Strategic Actions**: Concrete steps for implementing the recommendation.

Take into account your past mistakes on similar situations. Use these insights to refine your decision-making.

Here are your past reflections on mistakes:
\"{past_memory_str}\"

Here is the debate:
Debate History:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
