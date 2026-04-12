from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_fundamentals,
    get_stock_data,
    get_indicators,
    get_balance_sheet,
    get_income_statement,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_quant_analyst(llm):
    def quant_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        verified_data = state.get("verified_data", "")
        prefetched = state.get("prefetched_data") or {}

        tools = [
            get_stock_data,
            get_indicators,
            get_fundamentals,
            get_balance_sheet,
            get_income_statement,
        ]

        system_message = (
            "You are a QUANTITATIVE analyst. You work ONLY with numbers — no narratives, no stories, no qualitative opinions."
            "\n\nCRITICAL: If VERIFIED GROUND-TRUTH DATA is provided in the context, you MUST use those exact"
            " values for Current Price, 50-day SMA, 200-day SMA, RSI (14), 6-month return, ATR, Revenue Growth (YoY), and Earnings Growth (YoY)."
            " Do NOT recompute these metrics from raw data — use the verified figures as-is in your scorecard."
            " For all other metrics (P/E, EV/EBITDA, FCF Yield, etc.), use your tools normally."
            "\n\nSECTOR-SPECIFIC METRIC RULES (MANDATORY):"
            "\nBEFORE filling the Financial Health table, identify the company's GICS sector and sub-industry from the fundamentals data."
            " For the following sectors, standard ratios"
            " are structurally misleading and MUST be flagged:"
            "\n• INSURANCE: D/E is inflated by policyholder reserves (typical 10x-30x); report it but add"
            " '(insurance — reserves inflate D/E; not comparable to industrials)'. Current ratio is inapplicable."
            " Add combined ratio if available. Omit Current Ratio row or mark N/A."
            "\n• BANKING: D/E is inflated by deposits (typical 8x-12x); flag as"
            " '(bank — deposits are liabilities)'. Add CET1 or Tier 1 capital if available."
            "\n• REITs: Flag P/E as '(REIT — use P/FFO instead)'. High D/E is structural."
            "\n• UTILITIES: D/E of 1.5-3x is normal; flag the regulatory context."
            "\n• RESTAURANTS / FRANCHISORS: Mature franchise models (QSR, casual dining) routinely carry D/E of 3x-10x+ or even NEGATIVE equity due to aggressive share buybacks and dividend recapitalizations. MCD has negative equity; YUM has negative equity. A D/E of 3x at a franchisor like QSR/RBI is the INDUSTRY-STANDARD capital structure, not a red flag. Report D/E but add '(franchisor — high leverage is standard; compare to MCD, YUM, DPZ peers)'. Additionally, for margin comparisons: use OPERATING MARGIN, not net margin. Franchise revenue recognition includes system-wide sales that inflate the revenue base, making net margin appear artificially low vs non-franchise peers. Operating margin isolates the franchisor's actual profitability."
            "\n• CONSUMER STAPLES / TOBACCO: Stable cash flows support high leverage. D/E of 2x-5x is common (PM, MO, KO). Flag the sector norm."
            "\nIn the 'Sector Median' column of the Valuation table, use the ACTUAL sector — never generic S&P 500 medians"
            " for financials. If you don't have a reliable sector median, write 'N/A' — do NOT fabricate one."
            "\nIn Red Flags: NEVER flag D/E as 'concerning' or 'alarming' without comparing to SECTOR PEERS."
            "\nYour job is to produce a structured numerical scorecard for the given stock. You must:"
            "\n1. Retrieve the stock's price data for the last 6 months."
            "\n2. Retrieve key technical indicators: RSI (14-day), MACD (12,26,9), 50-day SMA, 200-day SMA."
            "\n3. Retrieve fundamental data: P/E ratio, EV/EBITDA, revenue growth (YoY), free cash flow yield, debt-to-equity."
            "\n4. Retrieve the balance sheet and income statement for the latest quarter."
            "\n\nYour output MUST follow this exact structure:"
            "\n"
            "\n## Quantitative Scorecard"
            "\n"
            "\n### Price & Technicals"
            "\n| Metric | Value |"
            "\n|--------|-------|"
            "\n| Current Price | $ |"
            "\n| 50-day SMA | $ |"
            "\n| 200-day SMA | $ |"
            "\n| Price vs 50-SMA | above/below by X% |"
            "\n| Price vs 200-SMA | above/below by X% |"
            "\n| RSI (14) | |"
            "\n| MACD Signal | bullish/bearish crossover |"
            "\n| ATR (14) | $X.XX |"
            "\n| ATR% (ATR/Price) | X.XX% (classify: very low <0.5%, low 0.5-1.5%, moderate 1.5-3%, high 3-5%, very high >5%) |"
            "\n| 2×ATR Stop | $X.XX (= Price − 2×ATR) |"
            "\n| Max Drawdown % (2×ATR) | X.X% (= 2×ATR / Price × 100 — if >15%, flag as WIDE STOP) |"
            "\n| 6-month return | X% |"
            "\n"
            "\n### Valuation"
            "\n| Metric | Value | Peer Median | Premium/Discount |"
            "\n|--------|-------|-------------|-----------------|"
            "\n| P/E (TTM) | | | +/-X% |"
            "\n| EV/EBITDA | | | +/-X% |"
            "\n| P/FCF | | | +/-X% |"
            "\n| PEG ratio | | | +/-X% |"
            "\nIMPORTANT: Use the INDUSTRY PEER COMPARISON data provided to fill the 'Peer Median'"
            " column with ACTUAL peer medians — NOT generic sector averages."
            " Compute Premium/Discount = (Target - Peer Median) / Peer Median × 100."
            " If peer data is unavailable, write 'N/A' — do NOT fabricate medians."
            "\n"
            "\n### Financial Health"
            "\n| Metric | Value |"
            "\n|--------|-------|"
            "\n| Revenue Growth (YoY) | (use VERIFIED value if provided — do NOT compute your own) |"
            "\n| FCF Yield | X% |"
            "\n| Debt-to-Equity | |"
            "\n| Current Ratio | |"
            "\n| Gross Margin | X% |"
            "\n| Operating Margin | X% |"
            "\n"
            "\n### Valuation Cross-Check"
            "\nDo NOT run your own DCF or DDM — the fundamentals analyst provides the"
            " intrinsic valuation with a sensitivity table. Your job is to VALIDATE, not duplicate."
            "\nUsing the financial data available, compute:"
            "\n1. The historical FCF CAGR (3-5 years) from the cash flow/income statements."
            "\n2. The historical Revenue CAGR (3-5 years) from the income statements."
            "\n3. Report these growth rates in the table below — they serve as a cross-check"
            " against the fundamentals analyst's growth assumptions."
            "\n| Metric | Value |"
            "\n|--------|-------|"
            "\n| Historical Revenue CAGR (3-5yr) | X% |"
            "\n| Historical FCF CAGR (3-5yr) | X% |"
            "\n| Forward EPS Implied Growth | X% (from trailing vs forward EPS if available) |"
            "\nDo NOT produce your own fair value estimate. The fundamentals analyst owns the DCF."
            "\n"
            "\n### Implied Expectations (MANDATORY — do this BEFORE your verdict)"
            "\nThis is the most important section. It asks: what does the current price ASSUME?"
            "\n1. **Implied EPS Growth**: At the current P/E of X, and trailing EPS of $Y,"
            "   what annual EPS growth rate does the market assume over 3-5 years to justify"
            "   this price? Use the PEG framework: if P/E = 40, the market expects ~40% annualized"
            "   earnings growth, or ~20% with a 2.0 PEG premium. Compare this to the ACTUAL"
            "   historical revenue/EPS growth rate from the financial statements."
            "\n2. **Growth Gap** = Implied Growth − Actual Historical Growth."
            "   INTERPRETATION (this is critical — do NOT invert the logic):"
            "   • If Gap is POSITIVE (Implied > Actual): the market expects MORE growth than the"
            "     company has historically delivered → stock is potentially OVERPRICED / priced for perfection."
            "     Example: Implied 30%, Actual 15% → Gap +15pp → market expects 2x historical growth."
            "   • If Gap is NEGATIVE (Implied < Actual): the market expects LESS growth than the"
            "     company has historically delivered → stock is potentially UNDERPRICED / market is conservative."
            "     Example: Implied 6%, Actual 15% → Gap -9pp → market under-prices the earnings trajectory."
            "   • If Gap is within ±3pp: FAIRLY PRICED relative to historical growth."
            "   A NEGATIVE gap is NOT a red flag for overvaluation — it is a potential buy signal."
            "\n3. **Reverse DCF**: At the current price, what FCF growth rate is implied?"
            "   Using base FCF from the financial statements and a 10% discount rate, what growth rate g makes"
            "   the DCF output equal X? If g > 20%, the stock assumes heroic growth."
            "\n4. **Extension Check**: If 6-month return > 50% AND price > 40% above 200-SMA,"
            "   flag as EXTENDED. A stock that has returned 80% in 6 months has likely already"
            "   priced in near-term catalysts. Buying after a massive run is chasing, not investing."
            "\n| Metric | Value |"
            "\n|--------|-------|"
            "\n| Implied EPS Growth (from P/E) | X% |"
            "\n| Actual Historical EPS Growth | X% |"
            "\n| Growth Gap (Implied − Actual) | +Xpp if Implied>Actual=OVERPRICED; −Xpp if Implied<Actual=UNDERPRICED; within ±3pp=FAIRLY PRICED |"
            "\n| Implied FCF Growth (reverse DCF) | X% |"
            "\n| 6-Month Return | X% |"
            "\n| Price vs 200-SMA | +X% |"
            "\n| Extension Flag | YES / NO |"
            "\n"
            "\n### Quantitative Verdict"
            "\nBased SOLELY on the numbers above (no narratives, no story):"
            "\n- **Signal**: BUY / SELL / HOLD"
            "\n- **Primary driver**: Which 1-2 metrics most strongly drive this signal"
            "\n- **Red flags**: Any metrics that are concerning regardless of the signal"
            "\n- **Implied growth assessment**: Is the market's implied growth rate achievable?"
            "\n"
            "\nDo NOT reference AI trends, management commentary, competitive moats, or any qualitative factors."
            "\nDo NOT read or reference any other analyst's report. Your analysis is numbers-only."
            + get_language_instruction(),
        )

        # Pre-fetched data mode: inject data, skip tool round-trips
        if prefetched.get("stock_data"):
            data_block = (
                "\n\n" + prefetched.get("company_profile", "")
                + "\n\n=== PRE-FETCHED STOCK PRICE DATA (OHLCV) ===\n"
                + prefetched["stock_data"]
                + "\n\n=== PRE-FETCHED TECHNICAL INDICATORS ===\n"
                + prefetched.get("indicators", "N/A")
                + "\n\n=== PRE-FETCHED FUNDAMENTALS ===\n"
                + prefetched.get("fundamentals", "N/A")
                + "\n\n=== PRE-FETCHED BALANCE SHEET (Quarterly) ===\n"
                + prefetched.get("balance_sheet", "N/A")
                + "\n\n=== PRE-FETCHED INCOME STATEMENT (Quarterly) ===\n"
                + prefetched.get("income_statement", "N/A")
                + "\n\n=== INDUSTRY PEER COMPARISON ===\n"
                + prefetched.get("peer_comps", "N/A")
            )
            mode_instruction = (
                "All required quantitative data has been pre-fetched"
                " and is provided below. Analyze the data directly to write your scorecard."
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
            return {"quant_report": result.content}

        # Fallback: tool-calling loop
        chain = prompt | llm.bind_tools(tools)

        # Internal tool loop — runs tool calls locally for parallel execution
        tool_map = {tool.name: tool for tool in tools}
        local_messages = list(state["messages"])

        for _ in range(10):
            result = chain.invoke(local_messages)
            if not result.tool_calls:
                return {"quant_report": result.content}
            local_messages.append(result)
            for tc in result.tool_calls:
                try:
                    tool_output = tool_map[tc["name"]].invoke(tc["args"])
                except Exception as e:
                    tool_output = f"Error: {e}"
                local_messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tc["id"])
                )

        return {"quant_report": result.content or ""}

    return quant_analyst_node
