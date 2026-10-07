"""
research_agent.py - the "brain" of the app.

What lives here:
  1. A web search tool (DuckDuckGo via the `ddgs` package, no API key needed)
  2. The Groq LLM connection
  3. One CrewAI Agent + one Task + one Crew
  4. Friendly error messages
  5. A check that compares URLs in the report against URLs the search tool really returned

app.py (the Streamlit UI) only calls run_research(topic).
"""

import os
import re
import time
from datetime import date

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from ddgs import DDGS
from dotenv import load_dotenv

# Reads the .env file when running locally. On Streamlit Cloud there is no
# .env file, and app.py copies the Streamlit secret into the environment instead.
load_dotenv()

# "groq/" tells CrewAI to use Groq (through LiteLLM).
# "openai/gpt-oss-120b" is Groq's own model ID.
MODEL_NAME = "groq/openai/gpt-oss-120b"

MAX_SEARCH_RESULTS = 5  # results per search
SNIPPET_LENGTH = 500  # characters kept per result (keeps prompts small -> fewer rate limits)


class ResearchError(Exception):
    """An error whose message is safe and friendly to show to the user."""


# ---------------------------------------------------------------------------
# 1. Web search tool
# ---------------------------------------------------------------------------
def make_search_tool(collected):
    """
    Build the search tool the agent can call.

    `collected` is a list that we fill with every real search result.
    Later we use it to (a) prove that a search really happened and
    (b) check that the URLs in the report are real.
    """

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web with DuckDuckGo. Input: a short search query.
        Returns numbered results with title, URL and snippet.
        If the search fails, the output starts with SEARCH_FAILED."""
        last_error = "unknown error"

        for _attempt in range(2):  # try twice, DuckDuckGo sometimes fails once
            try:
                results = DDGS().text(query, max_results=MAX_SEARCH_RESULTS)
            except Exception as e:  # network error, rate limit, blocked IP, ...
                last_error = f"{type(e).__name__}: {e}"
                time.sleep(2)
                continue

            lines = []
            for r in results or []:
                url = (r.get("href") or "").strip()
                if not url:
                    continue
                title = (r.get("title") or "").strip()
                snippet = (r.get("body") or "").strip()[:SNIPPET_LENGTH]

                if not any(item["url"] == url for item in collected):
                    collected.append({"title": title, "url": url})
                lines.append(f"[{len(lines) + 1}] {title}\nURL: {url}\nSnippet: {snippet}")

            if lines:
                return "\n\n".join(lines)

            last_error = "the search returned no results"
            break

        return (
            f"SEARCH_FAILED: {last_error}. "
            "No information was retrieved for this query. Do not invent results."
        )

    return web_search


# ---------------------------------------------------------------------------
# 2. Groq LLM
# ---------------------------------------------------------------------------
def get_llm():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ResearchError(
            "GROQ_API_KEY is missing. Locally: put it in your .env file. "
            "On Streamlit Cloud: add it under App settings -> Secrets."
        )
    return LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,  # low = more factual, less creative
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# 3. Error translation
# ---------------------------------------------------------------------------
def explain_error(error):
    """Turn a confusing exception into a message a beginner can act on."""
    text = f"{type(error).__name__}: {error}".lower()

    if "401" in text or "invalid api key" in text or "authentication" in text:
        return (
            "Groq rejected your API key (authentication error). "
            "Check that GROQ_API_KEY is correct, has no extra spaces or quotes, "
            "and has not been deleted in the Groq console."
        )
    if "429" in text or "rate limit" in text or "rate_limit" in text:
        return (
            "Groq rate limit reached. Wait about a minute and try again, "
            "or use a shorter, more focused topic."
        )
    if "model_not_found" in text or "does not exist" in text or "decommissioned" in text:
        return (
            f"Groq says the model is unavailable. The app is using '{MODEL_NAME}'. "
            "Check the model list in the Groq console and update MODEL_NAME in research_agent.py."
        )
    if "connection" in text or "timeout" in text or "timed out" in text or "network" in text:
        return "Network problem: could not reach Groq or the web. Check your internet connection and try again."
    if "litellm" in text:
        return (
            "CrewAI could not load its Groq connector (LiteLLM). "
            "Make sure you installed the requirements with: pip install -r requirements.txt"
        )
    return f"The research failed unexpectedly. Details: {type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
# 4. Source checking
# ---------------------------------------------------------------------------
def find_unverified_urls(report, collected):
    """Return URLs that appear in the report but were NOT returned by the search tool."""
    real = {item["url"].rstrip("/") for item in collected}
    found = re.findall(r"https?://[^\s)\]>\"']+", report)
    bad = []
    for url in found:
        clean = url.rstrip(".,;:").rstrip("/")
        if clean not in real and clean not in bad:
            bad.append(clean)
    return bad


# ---------------------------------------------------------------------------
# 5. The research function the UI calls
# ---------------------------------------------------------------------------
def run_research(topic):
    topic = (topic or "").strip()
    if not topic:
        raise ResearchError("Please enter a research topic first.")
    if len(topic) > 300:
        raise ResearchError("Please keep the topic under 300 characters.")

    llm = get_llm()
    collected = []
    search_tool = make_search_tool(collected)

    agent = Agent(
        role="Careful Research Analyst",
        goal=(
            "Produce an accurate, well-structured research report using ONLY "
            "information retrieved through the web_search tool."
        ),
        backstory=(
            "You are a meticulous analyst. You never state a fact you cannot trace to a "
            "search result, and you openly say when evidence is missing or weak."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=10,
    )

    task = Task(
        description=(
            "Research this topic: {topic}\n"
            "Today's date: {today}\n\n"
            "PROCESS:\n"
            "1. Use the web_search tool 2 to 4 times with different, specific queries.\n"
            "2. Read the results and compare what different sources say.\n"
            "3. Write the report.\n\n"
            "STRICT RULES:\n"
            "- Base every factual claim on the search results. Cite it as [1], [2] etc., "
            "matching the numbered Sources list at the end.\n"
            "- Never invent sources, URLs, statistics, studies, quotes or facts.\n"
            "- In the Sources section, list ONLY URLs that appeared in the tool output, "
            "copied exactly. Never write a URL you did not see in the tool output.\n"
            "- Never claim you searched a site or query that you did not search.\n"
            "- If the tool output starts with SEARCH_FAILED, say so plainly in the Limitations "
            "section. If all searches failed, write a short report stating that nothing "
            "could be verified.\n"
            "- Search snippets are short. Do not claim details beyond what a snippet says.\n"
            "- Prefer claims supported by more than one source, and say when only one source "
            "supports a claim.\n"
            "- Clearly mark anything you cannot verify with the phrase 'Not verified'.\n"
            "- Do not present assumptions as facts."
        ),
        expected_output=(
            "A markdown report with exactly these sections:\n"
            "# Research Report: <topic>\n"
            "## 1. Executive Summary\n"
            "## 2. Introduction\n"
            "## 3. Key Findings\n"
            "## 4. Detailed Analysis\n"
            "## 5. Important Facts and Evidence\n"
            "## 6. Different Perspectives\n"
            "## 7. Limitations (include what could not be verified)\n"
            "## 8. Conclusion\n"
            "## 9. Sources (numbered list: title and exact URL from the search results)"
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    try:
        result = crew.kickoff(inputs={"topic": topic, "today": date.today().isoformat()})
    except ResearchError:
        raise
    except Exception as e:
        raise ResearchError(explain_error(e)) from e

    report = str(getattr(result, "raw", result)).strip()

    # If no search ever returned anything, the report cannot be based on web research.
    if not collected:
        raise ResearchError(
            "The web search returned no results, so a report based on real sources "
            "could not be created. DuckDuckGo may be rate-limiting or blocking this "
            "network. Wait a minute and try again."
        )
    if not report:
        raise ResearchError("The model returned an empty report. Please try again.")

    return {
        "report": report,
        "sources": collected,  # every URL the search tool really returned
        "unverified_urls": find_unverified_urls(report, collected),
    }
