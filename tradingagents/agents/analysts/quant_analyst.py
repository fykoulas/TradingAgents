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
            "\n\nYour job is to produce a structured numerical scorecard for the given stock. You must:"
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
