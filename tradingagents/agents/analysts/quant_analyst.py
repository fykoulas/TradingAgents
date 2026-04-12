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
            " values for Current Price, 50-day SMA, 200-day SMA, RSI (14), 6-month return, and ATR."
            " Do NOT recompute these metrics from raw data — use the verified figures as-is in your scorecard."
            " For all other metrics (P/E, EV/EBITDA, Revenue Growth, FCF Yield, etc.), use your tools normally."
            "\n\nSECTOR-SPECIFIC METRIC RULES (MANDATORY):"
            "\nBEFORE filling the Financial Health table, identify the company's GICS sector from the fundamentals data."
            " For financial-sector firms (Insurance, Banks, Diversified Financials, REITs), standard ratios"
            " are structurally misleading and MUST be flagged:"
            "\n• INSURANCE: D/E is inflated by policyholder reserves (typical 10x-30x); report it but add"
            " '(insurance — reserves inflate D/E; not comparable to industrials)'. Current ratio is inapplicable."
            " Add combined ratio if available. Omit Current Ratio row or mark N/A."
            "\n• BANKING: D/E is inflated by deposits (typical 8x-12x); flag as"
            " '(bank — deposits are liabilities)'. Add CET1 or Tier 1 capital if available."
            "\n• REITs: Flag P/E as '(REIT — use P/FFO instead)'. High D/E is structural."
            "\n• UTILITIES: D/E of 1.5-3x is normal; flag the regulatory context."
            "\nIn the 'Sector Median' column of the Valuation table, use the ACTUAL sector — never generic S&P 500 medians"
            " for financials. If you don't have a reliable sector median, write 'N/A' — do NOT fabricate one."
            "\nIn Red Flags: NEVER flag a financial-sector D/E as 'concerning' without explaining the sector context."
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
            "\n| 6-month return | X% |"
            "\n"
            "\n### Valuation"
            "\n| Metric | Value | Sector Median |"
            "\n|--------|-------|--------------|"
            "\n| P/E (TTM) | | |"
            "\n| EV/EBITDA | | |"
            "\n| P/FCF | | |"
            "\n| PEG ratio | | |"
            "\n"
            "\n### Financial Health"
            "\n| Metric | Value |"
            "\n|--------|-------|"
            "\n| Revenue Growth (YoY) | X% |"
            "\n| FCF Yield | X% |"
            "\n| Debt-to-Equity | |"
            "\n| Current Ratio | |"
            "\n| Gross Margin | X% |"
            "\n| Operating Margin | X% |"
            "\n"
            "\n### Intrinsic Valuation Estimate"
            "\nCompute ONE simple valuation model using the financial data available:"
            "\n• For most sectors: Simple DCF — use the most recent annual FCF, a growth rate"
            "  derived from revenue/FCF trends (cap at 15%), terminal growth 2-3%, discount rate 10%."
            "\n• For Insurance/Banking: DDM — use trailing dividend, estimate sustainable growth"
            "  via g = ROE × (1 - payout ratio). Or P/Book relative to sector (life ~1.0x, P&C ~1.5x, banks ~1.0-1.5x)."
            "\n• For REITs: FFO-based yield model or NAV estimate."
            "\nShow your work:"
            "\n| Input | Value |"
            "\n|-------|-------|"
            "\n| Model | DCF / DDM / P/Book |"
            "\n| Base metric (FCF, Dividend, Book Value) | $ |"
            "\n| Growth rate assumed | X% |"
            "\n| Discount rate / required return | X% |"
            "\n| Estimated Fair Value / share | $ |"
            "\n| Current Price | $ |"
            "\n| Implied Upside/Downside | +/-X% |"
            "\nIf data is insufficient for any model, state what is missing — do NOT skip the section."
            "\n"
            "\n### Quantitative Verdict"
            "\nBased SOLELY on the numbers above (no narratives, no story):"
            "\n- **Signal**: BUY / SELL / HOLD"
            "\n- **Primary driver**: Which 1-2 metrics most strongly drive this signal"
            "\n- **Red flags**: Any metrics that are concerning regardless of the signal"
            "\n"
            "\nDo NOT reference AI trends, management commentary, competitive moats, or any qualitative factors."
            "\nDo NOT read or reference any other analyst's report. Your analysis is numbers-only."
            + get_language_instruction(),
        )

        # Pre-fetched data mode: inject data, skip tool round-trips
        if prefetched.get("stock_data"):
            data_block = (
                "\n\n=== PRE-FETCHED STOCK PRICE DATA (OHLCV) ===\n"
                + prefetched["stock_data"]
                + "\n\n=== PRE-FETCHED TECHNICAL INDICATORS ===\n"
                + prefetched.get("indicators", "N/A")
                + "\n\n=== PRE-FETCHED FUNDAMENTALS ===\n"
                + prefetched.get("fundamentals", "N/A")
                + "\n\n=== PRE-FETCHED BALANCE SHEET (Quarterly) ===\n"
                + prefetched.get("balance_sheet", "N/A")
                + "\n\n=== PRE-FETCHED INCOME STATEMENT (Quarterly) ===\n"
                + prefetched.get("income_statement", "N/A")
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
