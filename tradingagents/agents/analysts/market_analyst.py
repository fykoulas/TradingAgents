from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        verified_data = state.get("verified_data", "")
        prefetched = state.get("prefetched_data") or {}

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction. Tips: It lags price; combine with faster indicators for timely signals. NOTE: The 50-SMA is a TREND indicator, NOT a stop-loss level.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend direction. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries. NOTE: The 200-SMA is a TREND indicator, NOT a stop-loss level.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

ATR INTERPRETATION (MANDATORY when ATR is selected):
ATR in dollars is meaningless without context. You MUST compute and report:
  1. ATR% = ATR / Current Price × 100 (this is the daily volatility as a percentage)
  2. Classify volatility using ATR%:
     - < 0.5%: VERY LOW volatility (extremely tight range, typical of stable large-caps or low-liquidity ADRs)
     - 0.5%–1.5%: LOW volatility
     - 1.5%–3.0%: MODERATE volatility (normal for most liquid stocks)
     - 3.0%–5.0%: HIGH volatility
     - > 5.0%: VERY HIGH volatility (small-caps, biotechs, meme stocks)
  3. Compute actionable stop levels:
     - Swing stop (2× ATR): Current Price − (2 × ATR) = $X.XX
     - Wide stop (3× ATR): Current Price − (3 × ATR) = $X.XX
  4. NEVER call ATR 'moderate' or 'low' based on the dollar amount alone.
     $0.05 ATR sounds small but on a $15 stock it's 0.33% — very low.
     $5.00 ATR sounds large but on a $500 stock it's 1.0% — low.

STOP-LOSS RULE (MANDATORY):
Moving averages (50-SMA, 200-SMA) are TREND indicators — they tell you the direction,
not where to place your stop. NEVER recommend 'stop below 50-SMA' or 'stop below 200-SMA'
as a risk management level. The ONLY valid stop-loss methodology is ATR-based:
  - Use the 2×ATR or 3×ATR stop levels computed above.
  - SMA levels may be referenced for trend context (e.g., 'price is X% above 200-SMA'),
    but they are NOT stop-loss prices.
  - If the distance from entry to a SMA level exceeds 15% of the entry price, using that
    SMA as a 'stop' would produce a drawdown no professional risk manager would accept.
Your report must recommend exactly ONE stop level: the 2×ATR stop (for swing trades)
or the 3×ATR stop (for wider positions). Do NOT suggest multiple conflicting stop levels.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names. Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions.

PRICE EXTENSION ANALYSIS (MANDATORY):
After computing Price vs 200-SMA %, you MUST assess extension risk:
1. Compute the % distance from the 200-day SMA. If >40%, the stock is EXTENDED.
2. If the 6-month return is >50%, the stock has had an EXCEPTIONAL run.
3. When BOTH conditions are true (>40% above 200-SMA AND >50% 6-month return):
   - You MUST include a '### Price Extension Warning' section.
   - State explicitly: 'At X% above the 200-SMA with a Y% 6-month return, this stock
     is trading at levels that historically carry elevated mean-reversion risk.'
   - Do NOT treat momentum as a reason to buy. A stock that has already moved +80%
     may have already priced in the thesis. Momentum is descriptive, not predictive.
   - Ask: 'What further catalyst exists to justify buying AFTER an X% move?'
   - Note the gap between current price and 50-SMA as a pullback risk zone.
4. If Price vs 200-SMA > 20% but < 40%, note the extension but it is not yet extreme.
5. NEVER describe a stock >40% above its 200-SMA as having a 'healthy bullish trend.'
   It may be bullish, but it is NOT at a healthy entry point for new positions.
6. When RSI > 60 AND price > 40% above 200-SMA: the risk/reward for NEW entries is
   unfavorable. State this clearly. Existing positions may hold; new entries should wait."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        # Pre-fetched data mode: inject data, skip tool round-trips
        if prefetched.get("stock_data"):
            data_block = (
                "\n\n=== PRE-FETCHED STOCK PRICE DATA (OHLCV) ===\n"
                + prefetched["stock_data"]
                + "\n\n=== PRE-FETCHED TECHNICAL INDICATORS ===\n"
                + prefetched.get("indicators", "N/A")
            )
            mode_instruction = (
                "All required market data and technical indicators have been pre-fetched"
                " and are provided below. Analyze the data directly to write your report."
            )
        else:
            data_block = ""
            mode_instruction = (
                "Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK; another assistant with different tools"
                " will help where you left off. Execute what you can to make progress."
                f" You have access to the following tools: {', '.join([t.name for t in tools])}."
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " {mode_instruction}"
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}"
                    "\n\nCRITICAL — TODAY'S TRADING DATE IS {current_date}."
                    " All dates in your report MUST reference this exact date (year, month, day)."
                    " The last row in any OHLCV dataset is the most recent trading day"
                    " — use its Close as the current market price."
                    "\n\nIMPORTANT: If VERIFIED GROUND-TRUTH DATA is provided below, use those exact"
                    " values for RSI, 50-day SMA, 200-day SMA, ATR, and 6-month return in your report."
                    " Do NOT substitute tool-computed values that differ from the verified data."
                    " {instrument_context}\n\n{verified_data}"
                    "{data_block}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(mode_instruction=mode_instruction)
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(verified_data=verified_data)
        prompt = prompt.partial(
            data_block=data_block.replace("{", "{{").replace("}", "}}")
        )

        if data_block:
            # Single-shot: all data in prompt, no tools needed
            result = (prompt | llm).invoke(state["messages"])
            return {"market_report": result.content}

        # Fallback: tool-calling loop
        chain = prompt | llm.bind_tools(tools)

        # Internal tool loop — runs tool calls locally for parallel execution
        tool_map = {tool.name: tool for tool in tools}
        local_messages = list(state["messages"])

        for _ in range(10):
            result = chain.invoke(local_messages)
            if not result.tool_calls:
                return {"market_report": result.content}
            local_messages.append(result)
            for tc in result.tool_calls:
                try:
                    tool_output = tool_map[tc["name"]].invoke(tc["args"])
                except Exception as e:
                    tool_output = f"Error: {e}"
                local_messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tc["id"])
                )

        return {"market_report": result.content or ""}

    return market_analyst_node
