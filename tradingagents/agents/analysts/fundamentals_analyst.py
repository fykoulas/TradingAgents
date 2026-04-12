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
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            "\n\nCRITICAL: If VERIFIED GROUND-TRUTH DATA is provided in the context, you MUST use those exact"
            " values for Current Price and 6-Month Return. Do NOT compute your own price return from raw"
            " data or tool output - use the verified 6-Month Return figure as-is in your Reverse Valuation"
            " Check table. Computing your own return figure from partial data has produced errors exceeding"
            " 3x the actual value in prior analyses."
            "\n\nSECTOR-SPECIFIC INTERPRETATION (MANDATORY):"
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
            "\n• RESTAURANTS / FRANCHISORS: Mature franchise models (QSR, MCD, YUM, DPZ) routinely carry D/E of 3x-10x+ or even NEGATIVE equity. This is the standard capital structure for asset-light, cash-generative businesses that return capital aggressively. A D/E of 3x at a franchisor is NOT overleveraged — it reflects the business model. Compare to named peers (MCD, YUM, DPZ) before concluding leverage is excessive."
            "\n• CONSUMER STAPLES / TOBACCO: Stable, recurring cash flows support high leverage. D/E of 2x-5x is normal (PM, MO, KO). Flag sector context."
            " with predictable cash flows. Focus on interest coverage and regulatory rate-base growth."
            "\nIf you describe a ratio as 'conservative,' 'favorable,' 'low,' or 'concerning,'"
            " you MUST state what the relevant sector benchmark is and cite it."
            " Never call a metric 'low' or 'high' without stating the sector-appropriate range."
            "\n\nINTRINSIC VALUATION (MANDATORY):"
            "\nYou MUST include an absolute valuation estimate. A recommendation that rests"
            " only on relative multiples ('P/E looks cheap') is incomplete."
            "\n"
            "\nGROWTH RATE DERIVATION (do this FIRST before any model):"
            "\nYour growth assumption MUST be grounded in the company's actual financial history."
            " Do NOT pick a round number (4%, 5%, 10%) without justification."
            "\n1. Compute the company's historical revenue CAGR from the income statements (3-5 years)."
            "\n2. Compute historical FCF CAGR from cash flow statements (3-5 years)."
            "\n3. Note forward EPS estimates if available (trailing EPS vs forward EPS implies growth)."
            "\n4. Your BASE CASE growth rate = the lower of (historical revenue CAGR, historical FCF CAGR),"
            "   capped at 15%. Your BEAR case = Base × 0.5. Your BULL case = min(Base × 1.5, 20%)."
            "\n5. If you use a growth rate that differs from historical CAGR by more than 50%,"
            "   you MUST explain WHY (e.g., structural headwind, new product cycle, margin expansion)."
            "\n"
            "\nMODEL SELECTION by sector:"
            "\n• GENERAL (Tech, Industrials, Healthcare, Consumer): Simple DCF."
            "   Use FCF from the cash flow statement, discount at WACC 9-12% (10% default)."
            "\n• INSURANCE: DDM or Price/Book relative to ROE."
            "   g = ROE × (1 - payout ratio). Or P/Book vs sector (life ~1.0x, P&C ~1.5x)."
            "\n• BANKING: DDM or P/TBV relative to ROE. ROE > COE → P/TBV > 1.0x."
            "\n• REITs: P/FFO or P/AFFO vs. sector median. Estimate NAV if possible."
            "\n• UTILITIES: DDM using regulated return on rate base as growth proxy."
            "\n"
            "\nPresent a SENSITIVITY TABLE (not a single point estimate):"
            "\n| Scenario | Growth Rate | WACC/Discount | Fair Value | vs Current Price |"
            "\n|----------|-------------|---------------|------------|-----------------|"
            "\n| Bear     | X% (Base×0.5) | Y% | $XX.XX | +/-X% |"
            "\n| Base     | X% (from historicals) | Y% | $XX.XX | +/-X% |"
            "\n| Bull     | X% (Base×1.5, max 20%) | Y% | $XX.XX | +/-X% |"
            "\n"
            "\nAlso show your key inputs:"
            "\n| Input | Value | Source |"
            "\n|-------|-------|--------|"
            "\n| Model | DCF / DDM / P/Book |  |"
            "\n| Base FCF (or Dividend) | $X | Cash flow statement |"
            "\n| Historical Revenue CAGR | X% | Income statements |"
            "\n| Historical FCF CAGR | X% | Cash flow statements |"
            "\n| Forward EPS Growth (if available) | X% | Consensus |"
            "\n| Confidence | HIGH / MEDIUM / LOW |  |"
            "\n"
            "\nThe BASE CASE fair value is the primary reference for the rest of the report."
            " NEVER present a single point estimate without the sensitivity table."
            "\n\nREVERSE VALUATION CHECK (MANDATORY — do this AFTER your fair value estimate):"
            "\nAfter computing your intrinsic fair value, you MUST answer:"
            "\n1. At the CURRENT market price, what growth rate is the market implying?"
            "   Reverse your DCF/DDM: given Price = market price, solve for the growth rate g"
            "   that makes the model output equal the market price."
            "\n2. Compare the implied growth rate to the company's ACTUAL historical growth:"
            "   - What has revenue grown at over the past 3-5 years?"
            "   - What has EPS grown at over the past 3-5 years?"
            "\n3. If the implied growth rate EXCEEDS the historical rate by >50% (e.g., market"
            "   implies 30% growth but actual is 15%), the stock is PRICED FOR PERFECTION."
            "   Flag this prominently."
            "\n4. If the stock has returned >50% in 6 months, explicitly ask:"
            "   'Has the thesis already been PRICED IN by this move?' A stock that has already"
            "   appreciated 80% may have already captured the upside from the catalysts"
            "   the report identifies. Recommending BUY after a massive run requires proving"
            "   that significant ADDITIONAL upside exists beyond what has already been captured."
            "\n5. Present this in a clear table:"
            "\n| Check | Value |"
            "\n|-------|-------|"
            "\n| Market-Implied Growth (from reverse DCF/DDM) | X% |"
            "\n| Actual Historical Revenue Growth (3-5yr) | X% |"
            "\n| Actual Historical EPS Growth (3-5yr) | X% |"
            "\n| Growth Gap | Implied exceeds actual by X% — (reasonable / stretched / priced for perfection) |"
            "\n| 6-Month Return | (use VERIFIED value from ground-truth data — do NOT compute your own) |"
            "\n| Is thesis priced in? | YES / PARTIALLY / NO — with explanation |"
            "\n\nPEER COMPARISON (MANDATORY — use the INDUSTRY PEER COMPARISON data provided):"
            "\nIf peer comparison data is provided, you MUST include a '### Relative Valuation'"
            " section that:"
            "\n1. Names the 3-5 closest peers by market cap and business model."
            "\n2. Compares the target's P/E, Forward P/E, EV/EBITDA, and profit margins against"
            "   the peer group. Use the ACTUAL peer data provided — do NOT fabricate peer metrics."
            "\n3. States whether the target trades at a PREMIUM or DISCOUNT to peers, and by how much."
            "\n4. If the target trades at a premium, justify WHY it deserves it (faster growth,"
            "   better margins, dominant market position) with specific numbers."
            "\n5. If you cannot justify a premium and the target's multiples are above peer median,"
            "   flag it as a RELATIVE OVERVALUATION risk."
            "\n6. Present this as a table:"
            "\n| Metric | Target | Peer Median | Premium/Discount |"
            "\n|--------|--------|-------------|-----------------|"
            "\n| P/E (TTM) | | | +/-X% |"
            "\n| Fwd P/E | | | +/-X% |"
            "\n| EV/EBITDA | | | +/-X% |"
            "\n| Profit Margin | | | |"
            "\n\nSaying a stock is 'undervalued' or 'reasonably priced' without peer context is"
            " analytically empty. Every valuation judgment must be RELATIVE to comparable companies."
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
                + "\n\n=== INDUSTRY PEER COMPARISON ===\n"
                + prefetched.get("peer_comps", "N/A")
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
