from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction, get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        prefetched = state.get("prefetched_data") or {}

        tools = [
            get_news,
        ]

        system_message = (
            "You are a SENTIMENT ANALYST producing an institutional-grade sentiment report on a SPECIFIC company."
            " Your report must focus EXCLUSIVELY on sentiment, social media discussion, and public perception"
            " of THIS company. Every item you discuss must be about or directly relevant to this company."
            "\n\nFOCUS AREAS:"
            "\n• Company-specific news sentiment — analyst commentary, earnings reaction, management credibility"
            "\n• Social media discussion about THIS company — volume, tone, trending topics"
            "\n• Institutional sentiment signals — insider buying/selling, fund flow data, short interest changes"
            "\n• Retail investor sentiment — forums, social platforms, unusual options activity discussion"
            "\n\nSTRICT RELEVANCE FILTER:"
            "\nDo NOT include sentiment about unrelated companies, general market mood, or broad sector"
            " commentary unless it is specifically about this company's competitive position."
            " If the news data contains articles about other companies (Nvidia, Tesla, ETFs, etc.),"
            " IGNORE them entirely — they are noise."
            "\n\n=== ABSOLUTE PROHIBITION — NO FABRICATION ===\n"
            "If the data provided contains NO company-specific news or sentiment, or if the data"
            " explicitly states 'No news found' or 'DATA QUALITY: NO COMPANY-SPECIFIC NEWS FOUND',"
            " you MUST:\n"
            "  1. State: 'INSUFFICIENT DATA — No company-specific sentiment data available for [TICKER].'\n"
            "  2. Set your overall sentiment to: NEUTRAL — INSUFFICIENT DATA\n"
            "  3. Do NOT invent, infer, or speculate about what sentiment 'might be' or 'typically is'.\n"
            "  4. Do NOT describe what 'similar companies' or 'the sector' generally experiences.\n"
            "  5. Keep the report SHORT — a brief data-availability statement is better than fabrication.\n"
            "\n=== DATA PIPELINE FAILURE ESCALATION ===\n"
            "If — AND ONLY IF — the data below is tagged with the literal string\n"
            "'[DATA PIPELINE FAILURE' at the beginning, this means ALL search methods\n"
            "returned zero results for a company with significant market capitalisation.\n"
            "If articles or sentiment data ARE present in the data (even one), IGNORE this\n"
            "section entirely and analyze the data normally.\n"
            "When triggered, you MUST:\n"
            "  1. Open with: '⚠ **Data Quality Warning** — Sentiment data pipeline returned\n"
            "     zero results for a large-cap company. This is a data source failure.'\n"
            "  2. Set overall sentiment to: NEUTRAL — DATA PIPELINE FAILURE\n"
            "  3. State explicitly that sentiment assessment is UNRELIABLE and should not\n"
            "     influence the portfolio manager's decision.\n"
            "  4. Do NOT default to neutral-as-benign — absent data is a risk factor, not a\n"
            "     neutral signal.\n"
            "\n=== HEAVILY-COVERED COMPANY — PIPELINE LIMITATION RULE ===\n"
            "If the data below contains the tag '[SENTIMENT PIPELINE CONTEXT', this means\n"
            "the company is large-cap and/or has significant analyst coverage. For such\n"
            "companies, 'INSUFFICIENT DATA' is NEVER an acceptable conclusion when news\n"
            "articles are present. Instead, you MUST:\n"
            "  1. ANALYZE the sentiment of the news articles provided — analyst tone\n"
            "     (upgrades/downgrades, price target changes), earnings reaction, management\n"
            "     commentary, and institutional positioning signals.\n"
            "  2. If no social media-specific data (Reddit, Twitter, forums) exists, state:\n"
            "     'Social media feeds: NOT CONNECTED (pipeline limitation — not market silence).'\n"
            "  3. Your overall sentiment MUST be derived from the news articles — set it to\n"
            "     BULLISH, BEARISH, or NEUTRAL based on the evidence, NOT to 'INSUFFICIENT DATA'.\n"
            "  4. Include a row in your Summary Table:\n"
            "     'Social Media Feeds | N/A | No feeds connected (pipeline limitation) | N/A'\n"
            "  5. For a company with ≥5 analyst estimates or ≥$1B market cap, outputting\n"
            "     'INSUFFICIENT DATA' when news articles ARE available is a CRITICAL FAILURE.\n"
            "\nBANNED PHRASES (using any of these when you have no data is a CRITICAL FAILURE):\n"
            "  'typically', 'generally', 'likely', 'probably', 'usually', 'tends to',"
            "  'common for', 'one would expect', 'similar companies', 'in most cases',"
            "  'it is reasonable to assume', 'based on general trends'\n"
            "\nThese phrases signal fabrication. Every claim MUST cite a specific data point,"
            " article, or source from the provided data. No data = no claim."
            "\n\nOUTPUT STRUCTURE:"
            "\n1. **Sentiment Summary** — overall tone: bullish/bearish/neutral with evidence"
            "   (or INSUFFICIENT DATA if no company-specific data exists)"
            "\n2. **Key Sentiment Drivers** — specific events or discussions driving sentiment"
            "\n3. **Institutional vs Retail Divergence** — do institutions and retail agree?"
            "\n4. **Summary Table** — markdown table: Source/Signal | Sentiment | Key Detail | Reliability"
            + get_language_instruction()
        )

        # Pre-fetched data mode: inject data, skip tool round-trips
        if prefetched.get("company_news"):
            data_block = (
                "\n\n" + prefetched.get("company_profile", "")
                + "\n\n=== PRE-FETCHED COMPANY NEWS & SOCIAL MEDIA ===\n"
                + prefetched["company_news"]
            )
            mode_instruction = (
                "All required news and social media data has been pre-fetched"
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
                    " Do NOT output any BUY/SELL/HOLD recommendation or FINAL TRANSACTION PROPOSAL."
                    " Your role is to provide ANALYSIS ONLY — all trading decisions are made by code."
                    " At the END of your report, include a ```json assessment envelope with these exact fields:"
                    ' {{"sentiment": "BULLISH or BEARISH or NEUTRAL", "confidence": "HIGH or MEDIUM or LOW", "data_gaps": []}}'
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
            return {"sentiment_report": result.content}

        # Fallback: tool-calling loop
        chain = prompt | llm.bind_tools(tools)

        # Internal tool loop — runs tool calls locally for parallel execution
        tool_map = {tool.name: tool for tool in tools}
        local_messages = list(state["messages"])

        for _ in range(10):
            result = chain.invoke(local_messages)
            if not result.tool_calls:
                return {"sentiment_report": result.content}
            local_messages.append(result)
            for tc in result.tool_calls:
                try:
                    tool_output = tool_map[tc["name"]].invoke(tc["args"])
                except Exception as e:
                    tool_output = f"Error: {e}"
                local_messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tc["id"])
                )

        return {"sentiment_report": result.content or ""}

    return social_media_analyst_node
