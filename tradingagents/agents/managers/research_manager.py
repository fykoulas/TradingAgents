
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

**DATA COMPLETENESS CHECK (evaluate FIRST, before anything else):**
Scan the analyst reports below for any mention of:
- "[DATA QUALITY: ... UNAVAILABLE]" headers
- "No cash flow data available" or similar missing-data statements
- "INSUFFICIENT DATA" flags from any analyst
If ANY core financial statement (cash flow, balance sheet, income statement) is flagged
as missing or unavailable:
1. Acknowledge the gap explicitly in your recommendation.
2. A BUY recommendation is NOT permitted when critical financial data is missing.
   The maximum recommendation with material data gaps is HOLD.
3. If an analyst recommended BUY while ignoring a data gap, flag that as an analytical error.
4. State what data would need to become available before upgrading to BUY.

---

**SECTOR-AWARE RATIO INTERPRETATION:**
If the company is in Insurance, Banking, REITs, Restaurants/Franchisors, Utilities, or Consumer Staples/Tobacco, standard financial ratios (D/E, Current Ratio, P/E) are structurally different from industrial companies. Do NOT treat a high D/E at an insurer or bank as a red flag without acknowledging that policyholder reserves (insurance) or deposits (banking) inflate reported liabilities. Similarly, franchisors (QSR, MCD, YUM) routinely carry D/E of 3x-10x+ or negative equity — this is the standard franchise capital structure, not a crisis. If an analyst report flags D/E as 'alarming' or 'concerning' WITHOUT comparing to named sector peers, flag it as an analytical error in your evaluation.

---

**QUANTITATIVE REALITY CHECK (evaluate BEFORE reading the debate):**
Review the verified market data above. Before proceeding, answer:
- Is the stock above or below the 200-day SMA? By how much?
- What is the 6-month return?
- Is the RSI indicating oversold (<30), neutral (30-70), or overbought (>70)?
A BUY recommendation on a stock that is >10% below the 200-day SMA with no confirmed reversal signal requires EXCEPTIONAL justification beyond narrative conviction.

---

**PRICED-IN CHECK (evaluate BEFORE your recommendation):**
If the 6-month return is >50% AND the stock is near its 52-week high (within 10%):
1. You MUST explicitly ask: 'Has the bull thesis already been priced in by the recent move?'
2. A stock that has appreciated 80% in 6 months has likely already captured the near-term catalysts.
   Citing recent momentum or strong returns as reasons to BUY is CIRCULAR LOGIC.
   Momentum describes what ALREADY happened, not what will happen next.
3. To recommend BUY after a >50% 6-month run, you must identify SPECIFIC additional catalysts
   that have NOT yet been priced in, and quantify the remaining upside.
4. If the quant analyst's Implied Expectations section shows the market is pricing in growth
   that exceeds the company's historical rate, that is a red flag — not a buying opportunity.
5. Treat the fundamentals analyst's Reverse Valuation Check and the quant's Implied Growth Gap
   as key inputs. If both flag 'priced for perfection' or 'stretched,' BUY requires extraordinary
   justification (e.g., a specific earnings catalyst in the next 30 days with quantifiable impact).

---

**PEER COMPARISON CHECK:**
Scan the analyst reports for relative valuation against named industry peers.
If ANY analyst says the stock is 'undervalued,' 'reasonably priced,' or 'attractively valued'
without comparing to specific peers by name and multiple — flag that as an analytical weakness.
2. If peer comparisons use 'peer median' without NAMING the specific peers, flag this as
   incomplete. Every peer median must trace back to named companies listed in the peer data.
3. **Margin Metric Check (franchisors)**: If the company is a franchisor (QSR, MCD, YUM,
   DPZ, or classified as Restaurants/Fast Food), and any analyst compares NET profit margin
   to peers, flag this as WRONG METRIC. Franchise revenue recognition includes system-wide
   sales that inflate the revenue denominator. The correct comparison is OPERATING MARGIN.
   A net margin gap flagged as a 'weakness' for a franchisor is an analytical error.

---

**VERIFIED DATA CONSISTENCY CHECK:**
The VERIFIED GROUND-TRUTH DATA block contains independently computed values for Current Price,
6-Month Return, RSI, ATR, 52-Week High/Low, and SMA levels.
1. Scan ALL analyst reports for any reported 6-Month Return figure.
2. If ANY analyst's 6-Month Return differs from the VERIFIED value by more than 3 percentage
   points, flag it as a DATA ERROR. The verified value is authoritative — the divergent figure
   is hallucinated or computed from incomplete data.
3. If an analyst's conclusion depends on the wrong return figure (e.g., 'rapid appreciation
   suggests thesis is priced in' based on 55% when actual is 16%), flag the conclusion as
   INVALID and restate it using the verified figure.
4. Apply the same check to Current Price, RSI, and ATR — any divergence >5% is a flag.
A valuation judgment without peer context is incomplete.
If peer comparison data shows the target trading at a >20% PREMIUM to peer median P/E or EV/EBITDA,
your recommendation must acknowledge this and explain why the premium is justified.
If it cannot be justified, downgrade to HOLD.

---

**VALUATION CONSISTENCY CHECK:**
Scan ALL analyst reports for fair value estimates, DCF outputs, or intrinsic valuations.
1. List every fair value figure mentioned across all reports (fundamentals, quant, debate).
2. If ANY two fair value estimates differ by more than 15%, flag the conflict explicitly.
   State: 'Fundamentals estimates $X, Quant estimates $Y — these differ by Z%.'
3. Identify which growth rate assumption drives each estimate. If the growth rates differ,
   the estimates are not sensitivity scenarios — they are contradictory analyses.
4. Your recommendation must use the fundamentals analyst's BASE CASE fair value as the
   primary anchor. If any other section uses a different fair value, note the discrepancy.
5. If the fundamentals analyst's growth rate assumption deviates >50% from the company's
   historical revenue/FCF CAGR (e.g., using 4% growth when historicals show 10%), flag this
   as a potential underestimate or overestimate and state which direction.
6. **Growth Gap Direction Check**: Compare the quant's Implied EPS Growth vs Actual
   Historical EPS Growth against the fundamentals' Market-Implied Growth vs Actual
   Historical Revenue Growth. If one analyst concludes 'overpriced' (implied > actual)
   while the other concludes 'underpriced' or 'conservative' (implied < actual),
   flag the CONTRADICTION explicitly. State both figures, explain the discrepancy
   (different metrics — EPS vs revenue, P/E-implied vs DCF-implied), and state which
   conclusion is better supported by the underlying data. Do NOT let contradictory
   growth gap conclusions pass without reconciliation.

---

**STOP-LOSS CONSISTENCY CHECK:**
Scan the analyst reports for any stop-loss recommendations. Flag:
1. If ANY analyst suggests using a moving average (50-SMA, 200-SMA) as a stop-loss level,
   note this as an error. SMAs are trend indicators, not risk management tools.
2. If different analysts suggest different stop levels, list them and identify the conflict.
3. The correct stop methodology is ATR-based (2×ATR from the quant scorecard).
   Reference the quant’s 2×ATR Stop price as the authoritative stop level.
4. If the 2×ATR stop implies a drawdown >15% from entry, flag this: the stock may be
   too volatile for standard position sizing, or a tighter ATR multiple is needed.

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
