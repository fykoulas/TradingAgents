from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        prefetched = state.get("prefetched_data") or {}

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "You are a NEWS ANALYST producing an institutional-grade research note on a SPECIFIC company."
            " Your report must be directly actionable for a portfolio manager deciding whether to"
            " buy, hold, or sell THIS stock. Every paragraph must connect back to the company under review."
            "\n\nCOMPANY NEWS (PRIMARY — 60% of report):"
            "\nAnalyze the company-specific news. Cover: earnings/guidance updates, management changes,"
            " M&A activity, regulatory actions, product launches, litigation, analyst upgrades/downgrades,"
            " and material insider activity. If no company-specific news exists, state that explicitly —"
            " do NOT substitute unrelated stories."
            "\n\nSECTOR & MACRO NEWS (SECONDARY — 40% of report):"
            "\nFrom the global/macro news provided, include ONLY items that have a DIRECT, IDENTIFIABLE"
            " impact pathway to this company. You must explain HOW each macro item affects THIS company."
            " Consider the company's specific exposures:"
            "\n• GEOGRAPHIC: Where does the company earn revenue? FX risk for ADRs, currency exposure"
            "  for multinationals, country-specific regulatory changes."
            "\n• SECTOR: Central bank policy relevant to the company's sector (e.g. rate sensitivity for"
            "  banks/insurers/REITs, commodity prices for energy/materials, regulatory changes)."
            "\n• SUPPLY CHAIN: Input cost changes, trade policy, tariffs affecting the company's operations."
            "\n• COMPETITIVE: News about direct competitors, not tangentially related tech/market stories."
            "\n\nSTRICT RELEVANCE FILTER:"
            "\nDo NOT include news about unrelated companies (Nvidia, Tesla, AMD, etc.) unless they are"
            " a direct competitor, customer, or supplier of the company under review."
            " Do NOT include generic 'stock market roundup' or 'ETF flow' stories."
            " Do NOT pad the report with irrelevant macro commentary."
            " If a macro item cannot be connected to THIS company in one sentence, EXCLUDE it."
            "\n\n=== ABSOLUTE PROHIBITION — NO FABRICATION ===\n"
            "If the company news data states 'No news found' or 'DATA QUALITY: NO COMPANY-SPECIFIC"
            " NEWS FOUND', you MUST:\n"
            "  1. State: 'No company-specific news flow detected for [TICKER] in the review period.'\n"
            "  2. Do NOT fill the report with unrelated stories to compensate for missing data.\n"
            "  3. The Company-Specific Developments section should say 'None detected' — not be"
            "     replaced with Nvidia, Tesla, ETF, or generic market articles.\n"
            "  4. A short, honest report is ALWAYS better than a long fabricated one.\n"
            "\n=== DATA PIPELINE FAILURE ESCALATION ===\n"
            "If the data is tagged [DATA PIPELINE FAILURE], this means BOTH ticker-based AND\n"
            "company-name searches returned zero results for a company with significant market\n"
            "capitalisation. This is almost certainly a data source failure, NOT genuine silence.\n"
            "You MUST:\n"
            "  1. Open with a clear **⚠ Data Quality Warning** section BEFORE any analysis.\n"
            "  2. State: 'NEWS DATA UNAVAILABLE — data pipeline returned zero articles for a\n"
            "     [market cap] company. This is a data source failure, not evidence of market\n"
            "     silence. All news-dependent conclusions in this report are UNRELIABLE.'\n"
            "  3. Do NOT provide a bullish/bearish recommendation based on absent data.\n"
            "  4. Your Summary Table must include a row: 'Data Pipeline Failure | N/A | Cannot\n"
            "     assess news-driven risk | INCONCLUSIVE'\n"
            "  5. End with: 'RECOMMENDATION IMPACT: News assessment is INCOMPLETE. Portfolio\n"
            "     manager should verify news flow through an independent source before acting.'\n"
            "\nBANNED PHRASES (using these when you have no company data is a CRITICAL FAILURE):\n"
            "  'typically', 'generally', 'likely', 'probably', 'usually', 'tends to',"
            "  'common for', 'one would expect', 'similar companies', 'in most cases',"
            "  'it is reasonable to assume', 'based on general trends'\n"
            "\nEvery claim MUST cite a specific article, data point, or source. No data = no claim."
            "\n\nOUTPUT STRUCTURE:"
            "\n1. **Company-Specific Developments** — material news about the company itself"
            "   (or 'None detected in review period' if no company news exists)"
            "\n2. **Sector-Relevant Macro** — macro/regulatory items WITH explicit impact on this company"
            "\n3. **Key Risks from News Flow** — what new risks has the news revealed?"
            "\n4. **Summary Table** — markdown table: News Item | Source | Impact on Company | Bullish/Bearish"
            + get_language_instruction()
        )

        # Pre-fetched data mode: inject data, skip tool round-trips
        if prefetched.get("company_news"):
            data_block = (
                "\n\n" + prefetched.get("company_profile", "")
                + "\n\n=== PRE-FETCHED COMPANY NEWS ===\n"
                + prefetched["company_news"]
                + "\n\n=== PRE-FETCHED GLOBAL / MACRO NEWS ===\n"
                + prefetched.get("global_news", "N/A")
            )
            mode_instruction = (
                "All required news data has been pre-fetched"
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
            return {"news_report": result.content}

        # Fallback: tool-calling loop
        chain = prompt | llm.bind_tools(tools)

        # Internal tool loop — runs tool calls locally for parallel execution
        tool_map = {tool.name: tool for tool in tools}
        local_messages = list(state["messages"])

        for _ in range(10):
            result = chain.invoke(local_messages)
            if not result.tool_calls:
                return {"news_report": result.content}
            local_messages.append(result)
            for tc in result.tool_calls:
                try:
                    tool_output = tool_map[tc["name"]].invoke(tc["args"])
                except Exception as e:
                    tool_output = f"Error: {e}"
                local_messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tc["id"])
                )

        return {"news_report": result.content or ""}

    return news_analyst_node
