from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        prefetched = state.get("prefetched_data") or {}

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."            "\n\nSECTOR-SPECIFIC INTERPRETATION (MANDATORY):"
            "\nBEFORE interpreting ANY financial ratio, FIRST identify the company's GICS sector and industry."
            " Standard industrial-company benchmarks DO NOT apply to financial-sector firms:"
            "\n• INSURANCE: Debt-to-equity is structurally misleading — policyholder reserves and float"
            " are booked as liabilities, inflating D/E to 10x-30x+ even for healthy insurers."
            " A D/E of 20-30 does NOT mean 'overleveraged.' Use combined ratio (<100 = underwriting"
            " profit), investment yield, solvency margin, and reserve adequacy instead."
            " Current ratio is also inapplicable — insurers match long-tail liabilities, not short-term liquidity."
            "\n• BANKING: Deposits are liabilities, making D/E meaningless (typical 8x-12x)."
            " Use CET1 capital ratio (>10% = well-capitalised), net interest margin,"
            " non-performing loan ratio, and efficiency ratio instead."
            "\n• REITs: P/E is distorted by depreciation — use P/FFO or P/AFFO."
            " High D/E is structural (leveraged real assets with stable cash flows)."
            " Use debt/total-assets, interest coverage ratio, and occupancy rate."
            "\n• UTILITIES: High D/E (1.5x-3x) is normal for regulated, capital-intensive businesses"
            " with predictable cash flows. Focus on interest coverage and regulatory rate-base growth."
            "\nIf you describe a ratio as 'conservative,' 'favorable,' 'low,' or 'concerning,'"
            " you MUST state what the relevant sector benchmark is and cite it."
            " Never call a metric 'low' or 'high' without stating the sector-appropriate range."
            "\n\nINTRINSIC VALUATION (MANDATORY):"
            "\nYou MUST include at least ONE absolute valuation estimate. A recommendation that rests"
            " only on relative multiples ('P/E looks cheap') is incomplete. Choose the model"
            " appropriate to the sector:"
            "\n• GENERAL (Tech, Industrials, Healthcare, Consumer): Simple DCF."
            " Use FCF from the cash flow statement, apply a conservative growth rate derived from"
            " recent revenue/FCF trends (do NOT assume >15% perpetual growth), and discount at"
            " WACC of 9-12% (or 10% if data is insufficient). Show your inputs explicitly:"
            " base FCF, growth rate, terminal growth (2-3%), discount rate, resulting fair value per share."
            "\n• INSURANCE: Dividend Discount Model (DDM) or Price/Book relative to ROE."
            " Insurers return capital via dividends; use trailing dividend, payout ratio, and ROE"
            " to estimate sustainable dividend growth: g = ROE × (1 - payout ratio)."
            " If dividend data is unavailable, use P/Book vs. sector median P/Book (life ~1.0x, P&C ~1.5x)."
            " Also note Price/Embedded Value if data is available."
            "\n• BANKING: DDM or P/TBV (Price to Tangible Book Value) relative to ROE."
            " Banks with ROE > cost of equity deserve P/TBV > 1.0x. Show the comparison."
            "\n• REITs: P/FFO or P/AFFO vs. sector median. Estimate NAV if possible."
            "\n• UTILITIES: DDM using regulated return on rate base as growth proxy."
            "\nPresent your valuation estimate in a clear table:"
            "\n| Input | Value |"
            "\n|-------|-------|"
            "\n| Model Used | DCF / DDM / P/Book ... |"
            "\n| Key Inputs | (list each) |"
            "\n| Estimated Fair Value | $X.XX per share |"
            "\n| Current Price | $X.XX |"
            "\n| Upside/Downside | +/-X% |"
            "\nState your confidence in the estimate (HIGH/MEDIUM/LOW) and the biggest sensitivity"
            " (which input, if changed by 20%, would most affect the result)."
            "\n\nMATERIAL DATA GAPS (MANDATORY ESCALATION):"
            "\nIf ANY core financial statement (cash flow, balance sheet, income statement) is"
            " missing or marked [DATA QUALITY: ... UNAVAILABLE], you MUST:"
            "\n1. Flag it prominently in a dedicated '### Material Data Gaps' section at the TOP"
            "   of your report, before any analysis."
            "\n2. Explain WHY the missing data matters for this specific sector/company."
            "   For insurers: operating cash flow is critical because IFRS 17 / GAAP insurance"
            "   accounting makes net income unreliable — cash flow reveals true earnings quality."
            "   For REITs: FFO/AFFO from cash flow is the primary valuation metric."
            "   For all sectors: cash flow is required for DCF valuation."
            "\n3. Cap your confidence at LOW if cash flow data is missing."
            "\n4. Do NOT recommend BUY with missing critical data. The maximum recommendation"
            "   when a core financial statement is unavailable is HOLD, with a note that the"
            "   position should be revisited when complete data becomes available."
            "\n5. Do NOT paper over the gap with phrases like 'despite limited data' or"
            "   'this limitation should be considered' — these are weasel phrases that allow"
            "   the report to proceed as if the gap doesn't matter. It does."            + get_language_instruction(),
        )

        # Pre-fetched data mode: inject data, skip tool round-trips
        if prefetched.get("fundamentals"):
            data_block = (
                "\n\n=== PRE-FETCHED FUNDAMENTALS ===\n"
                + prefetched["fundamentals"]
                + "\n\n=== PRE-FETCHED BALANCE SHEET (Quarterly) ===\n"
                + prefetched.get("balance_sheet", "N/A")
                + "\n\n=== PRE-FETCHED CASH FLOW STATEMENT (Quarterly) ===\n"
                + prefetched.get("cashflow", "N/A")
                + "\n\n=== PRE-FETCHED INCOME STATEMENT (Quarterly) ===\n"
                + prefetched.get("income_statement", "N/A")
            )
            mode_instruction = (
                "All required fundamental data has been pre-fetched"
                " and is provided below. Analyze the data directly to write your report."
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
                    " {instrument_context}"
                    "{data_block}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(mode_instruction=mode_instruction)
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(
            data_block=data_block.replace("{", "{{").replace("}", "}}")
        )

        if data_block:
            # Single-shot: all data in prompt, no tools needed
            result = (prompt | llm).invoke(state["messages"])
            return {"fundamentals_report": result.content}

        # Fallback: tool-calling loop
        chain = prompt | llm.bind_tools(tools)

        # Internal tool loop — runs tool calls locally for parallel execution
        tool_map = {tool.name: tool for tool in tools}
        local_messages = list(state["messages"])

        for _ in range(10):
            result = chain.invoke(local_messages)
            if not result.tool_calls:
                return {"fundamentals_report": result.content}
            local_messages.append(result)
            for tc in result.tool_calls:
                try:
                    tool_output = tool_map[tc["name"]].invoke(tc["args"])
                except Exception as e:
                    tool_output = f"Error: {e}"
                local_messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tc["id"])
                )

        return {"fundamentals_report": result.content or ""}

    return fundamentals_analyst_node
