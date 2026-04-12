
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

**D/E DECOMPOSITION CHECK (MANDATORY — applies to ALL sectors):**
Scan the verified data for debt decomposition fields (Total Financial Debt, Net Debt/Cash,
Equity Ratio, Accumulated Deficit). Also check for a THIN-EQUITY ALERT.

If the Equity Ratio (Equity/Assets) is <10%:
1. The D/E ratio is MATHEMATICALLY UNINFORMATIVE — it is extreme because equity is near zero
   from accumulated operating losses, NOT necessarily because debt is catastrophic.
2. If ANY analyst describes the D/E as 'extremely high leverage,' 'alarming debt,' or
   'overleveraged' without decomposing it into Total Financial Debt vs Total Cash,
   flag this as an ANALYTICAL ERROR. The correct framing is: 'thin equity cushion' or
   'near-insolvency by book value' — these are different risks with different implications.
3. If the company has a NET CASH position (Total Cash > Total Debt) but an extreme D/E,
   any analyst treating D/E as a leverage alarm is WRONG. Net cash = no financial distress
   from debt, regardless of what D/E shows.
4. Check that the fundamentals analyst included the debt decomposition table. If missing,
   note it as a gap.
5. The correct risk to flag for thin-equity companies is: 'Company has almost no book equity
   cushion — a modest adverse event could make equity negative.' NOT: 'Company has extreme debt.'

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
4. **Peer Size Relevance**: Check whether the peer data includes a PEER SIZE MISMATCH or
   PEER SIZE GAP warning. If it does:
   a) Scan the analyst reports — did they acknowledge the size gap? If any analyst used
      peer multiples as-is without noting the market cap disparity, flag as ANALYTICAL WEAKNESS.
   b) If median peer market cap is >10x the target, peer-based 'undervalued' or 'overvalued'
      conclusions have LOW CONFIDENCE. Larger companies trade at higher multiples due to
      liquidity, coverage, and lower risk. A small-cap 'discount' to large-cap peers may
      simply reflect the appropriate size premium, NOT undervaluation.
   c) Check: did the fundamentals analyst identify any ACTUAL business competitors (same
      product market, similar revenue scale) beyond the classification peers? If not, note
      that no genuine comparables were identified and peer-based valuation is unreliable.
   d) If peer relevance is LOW and the recommendation relies heavily on relative valuation
      ('undervalued vs peers'), flag that the thesis foundation is weak.

---

**VERIFIED DATA CONSISTENCY CHECK:**
The VERIFIED GROUND-TRUTH DATA block contains independently computed values for Current Price,
6-Month Return, RSI, ATR, 52-Week High/Low, and SMA levels.

**CURRENT RATIO DECOMPOSITION CHECK:**
If the verified data includes a CURRENT LIABILITY DECOMPOSITION section (triggered when CR < 1.0):
1. Check whether the fundamentals analyst decomposed the current ratio or just flagged it
   as 'below optimal.' If any analyst says CR is 'concerning,' 'below the benchmark,' or
   'indicates liquidity risk' WITHOUT checking the deferred revenue composition, flag as
   ANALYTICAL ERROR. For SaaS/subscription businesses, deferred revenue is often 30-60%
   of current liabilities and is NOT a cash obligation.
2. If the verified data shows an Adjusted CR (excl. deferred revenue) >= 1.0, any report
   treating CR < 1.0 as a solvency risk is WRONG. The correct framing is: 'CR is depressed
   by deferred revenue; adjusted CR of X.XX shows adequate liquidity.'
3. If BOTH raw and adjusted CR are < 1.0, validate that the analyst checked: cash position,
   current debt amount, and whether credit facilities exist.

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

**FORWARD P/E CONTRADICTION CHECK (MANDATORY):**
Scan ALL analyst reports for conflicting valuation conclusions:
1. If the fundamentals analyst's peer table shows the target trading at a FORWARD P/E PREMIUM
   to peers (target Fwd P/E > peer median Fwd P/E), but any section of the report (Bull Thesis,
   your own recommendation) characterises the stock as 'undervalued compared to peers' or
   'attractively priced,' flag this as a CONTRADICTION.
2. A stock trading at a forward P/E premium is MORE EXPENSIVE than peers on expected earnings —
   it can only be 'undervalued' if a DIFFERENT metric (e.g., EV/Revenue, operating margin
   discount) is explicitly cited AND that metric is argued to be more relevant than forward P/E.
3. If trailing EPS is negative but forward P/E exists, the forward P/E is based on TURNAROUND
   expectations. The analyst must have stated the source and credibility of the forward EPS
   estimate. If this is missing, flag it: 'Forward P/E of X is cited without EPS provenance.'
4. Check analyst coverage count from verified data. If <5 analysts and the recommendation leans
   heavily on forward P/E, flag: 'Forward valuation anchor is based on thin analyst coverage.'
5. Reconcile the contradiction explicitly: state which metric supports 'undervalued,' which
   supports 'premium,' and WHICH ONE YOU WEIGHT MORE and why.
   Generic 'undervalued' without resolving the forward P/E premium is an ANALYTICAL ERROR.

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

**FCF RELIABILITY CHECK (MANDATORY):**
Scan the verified data for FCF DECOMPOSITION fields. If present, check the following:
1. **Base FCF Source**: The fundamentals analyst MUST have used the TTM quarterly sum as
   the base FCF (NOT info.freeCashflow) when the verified data flags a FCF DISCREPANCY.
   If the analyst's base FCF matches info.freeCashflow instead of TTM quarterly sum, flag
   as ANALYTICAL ERROR — the analyst used a stale/incorrect aggregate figure.
2. **FCF Volatility**: If the verified data shows FCF CV > 0.5 (HIGH FCF VOLATILITY),
   check that the fundamentals analyst either: (a) used median quarterly FCF × 4 as base,
   (b) applied wider sensitivity scenarios, or (c) switched to a revenue-based model.
   A single-point DCF with volatile FCF and narrow scenarios is UNRELIABLE.
3. **Turnaround Model Selection**: If the verified data flags TURNAROUND ALERT (negative
   trailing EPS + positive FCF), the fundamentals analyst should NOT have used a standard
   FCF-based DCF. Check that a revenue-based or earnings-power model was used instead.
   If the analyst used a DCF with turnaround FCF data, the valuation is UNRELIABLE —
   flag it and state that the recommendation cannot rely on the DCF fair value.
4. **FCF Reconciliation Table**: Check that the fundamentals report includes a reconciliation
   table showing TTM FCF, info.freeCashflow, discrepancy %, quarterly CV, base FCF used,
   and model chosen. If missing, flag as an analytical gap.

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

1. **Upstream Signal Tally (count FIRST)**:
   Scan ALL analyst reports and the debate for directional signals. Tally them explicitly:
   | Source | Signal |
   |--------|--------|
   | Quant Scorecard | BUY / HOLD / SELL |
   | Market Analysis | BUY / HOLD / SELL |
   | Fundamentals Analyst | BUY / HOLD / SELL |
   | Bull Thesis conclusion | strong / moderate / weak |
   | Bear Thesis conclusion | strong / moderate / weak |
   State the tally: 'Upstream consensus: X SELL, Y HOLD, Z BUY.'
   This tally constrains your recommendation — see Override Rules below.

2. **Strongest 3 arguments AGAINST your recommendation**: List the most compelling reasons from the opposing side of the debate. Do not dismiss them — steel-man them.
3. **What would change your mind**: State 1-2 specific, measurable conditions that would flip your recommendation (e.g., "If revenue growth drops below 10% next quarter" or "If the stock breaks below $X support").
4. **Confidence qualifier**: Rate your confidence as HIGH (>80% the debate evidence clearly favours one side), MEDIUM (60-80% evidence is mixed but leans), or LOW (<60% evidence is genuinely ambiguous). If LOW, your recommendation MUST be Hold.

**CONSENSUS OVERRIDE RULES:**
If ≥3 upstream signals agree on a direction (e.g., 3+ SELL), you MUST NOT override that
consensus with a contrary recommendation unless ALL of the following are met:
a) You cite a specific, quantified fair value and compute the risk/reward from current price.
b) You identify a concrete catalyst with a timeline (not speculative language like 'could
   improve,' 'might stabilize,' 'potential partnerships').
c) RSI between 30-40 is NOT 'oversold' — only RSI <30 qualifies. Do NOT use 'approaching
   oversold' as a reason to override a SELL consensus.
d) You perform a probability-weighted expected value calculation showing positive EV.
If you cannot satisfy (a)-(d), you MUST align with the upstream consensus.

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
