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

import functools
import inspect
import os
import re
import tempfile
import time
import traceback
from datetime import date

# These MUST be set before crewai is imported.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# Some hosts (e.g. Streamlit Cloud) only allow writing to the temp folder.
os.environ.setdefault("CREWAI_STORAGE_DIR", os.path.join(tempfile.gettempdir(), "crewai_storage"))

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402
from ddgs import DDGS  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Reads the .env file when running locally. On Streamlit Cloud there is no
# .env file, and app.py copies the Streamlit secret into the environment instead.
load_dotenv()


# ---------------------------------------------------------------------------
# 0. Compatibility fix: Groq rejects the "cache_breakpoint" field
# ---------------------------------------------------------------------------
# Newer CrewAI versions attach a "cache_breakpoint" key to messages (prompt caching
# for providers like Anthropic). LiteLLM forwards it to Groq, and Groq answers with
# "property 'cache_breakpoint' is unsupported". We remove the key right before the call.
def _strip_unsupported_keys(messages):
    cleaned = []
    for message in messages:
        if isinstance(message, dict):
            message = {k: v for k, v in message.items() if k != "cache_breakpoint"}
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [
                    {k: v for k, v in block.items() if k != "cache_breakpoint"}
                    if isinstance(block, dict)
                    else block
                    for block in content
                ]
        cleaned.append(message)
    return cleaned


def _fix_call_arguments(args, kwargs):
    if isinstance(kwargs.get("messages"), list):
        kwargs["messages"] = _strip_unsupported_keys(kwargs["messages"])
    elif len(args) > 1 and isinstance(args[1], list):  # completion(model, messages, ...)
        args = (args[0], _strip_unsupported_keys(args[1])) + tuple(args[2:])
    return args, kwargs


def _patch_litellm_for_groq():
    try:
        import litellm
    except Exception:
        return  # LiteLLM missing: nothing to patch (the real error will surface later)
    if getattr(litellm, "_groq_cache_fix_applied", False):
        return  # already patched (Streamlit re-runs the script many times)

    for name in ("completion", "acompletion"):
        original = getattr(litellm, name, None)
        if original is None:
            continue

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def wrapper(*args, __orig=original, **kwargs):
                args, kwargs = _fix_call_arguments(args, kwargs)
                return await __orig(*args, **kwargs)

        else:

            @functools.wraps(original)
            def wrapper(*args, __orig=original, **kwargs):
                args, kwargs = _fix_call_arguments(args, kwargs)
                return __orig(*args, **kwargs)

        setattr(litellm, name, wrapper)

    litellm._groq_cache_fix_applied = True


_patch_litellm_for_groq()

# "groq/" tells CrewAI to use Groq (through LiteLLM).
# "openai/gpt-oss-120b" is Groq's own model ID.
MODEL_NAME = "groq/openai/gpt-oss-120b"

MAX_SEARCH_RESULTS = 5  # results per search
SNIPPET_LENGTH = 400  # characters kept per result (small prompts -> fewer rate-limit errors)
RATE_LIMIT_WAIT_SECONDS = 25  # pause before the one automatic retry


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
        query = str(query or "").strip()
        if not query:
            return "SEARCH_FAILED: the query was empty. No information was retrieved."

        for attempt in range(2):  # try twice, DuckDuckGo sometimes fails once
            try:
                results = DDGS().text(query, max_results=MAX_SEARCH_RESULTS)
            except Exception as e:  # network error, rate limit, blocked IP, ...
                last_error = f"{type(e).__name__}: {e}"
                if attempt == 0:
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
    api_key = os.getenv("GROQ_API_KEY", "").strip().strip("\"'")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ResearchError(
            "GROQ_API_KEY is missing. Locally: put it in your .env file. "
            "On Streamlit Cloud: add it under App settings -> Secrets."
        )

    settings = dict(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,  # low = more factual, less creative
        # gpt-oss is a reasoning model: its hidden "thinking" tokens count toward this
        # limit, so it must be generous or the report can come back cut off or empty.
        max_tokens=6000,
        timeout=120,
    )
    try:
        # "low" keeps thinking short: faster, and uses less of Groq's tokens-per-minute budget.
        return LLM(reasoning_effort="low", **settings)
    except TypeError:
        # Older CrewAI versions do not know this option.
        return LLM(**settings)


# ---------------------------------------------------------------------------
# 3. Error translation
# ---------------------------------------------------------------------------
def is_rate_limit(error):
    text = f"{type(error).__name__}: {error}".lower()
    return bool(re.search(r"\b429\b", text)) or "rate limit" in text or "rate_limit" in text


def explain_error(error):
    """Turn a confusing exception into a message a beginner can act on."""
    text = f"{type(error).__name__}: {error}".lower()

    if re.search(r"\b401\b", text) or "invalid api key" in text or "authentication" in text:
        return (
            "Groq rejected your API key (authentication error). "
            "Check that GROQ_API_KEY is correct, has no extra spaces or quotes, "
            "and has not been deleted in the Groq console."
        )
    if is_rate_limit(error):
        return (
            "Groq rate limit reached. Wait about a minute and try again, "
            "or use a shorter, more focused topic."
        )
    if re.search(r"\b413\b", text) or "request too large" in text or "tokens per minute" in text:
        return (
            "The request was too large for Groq's per-minute token limit. "
            "Wait a minute and try again with a shorter, more focused topic."
        )
    if "model_not_found" in text or "does not exist" in text or "decommissioned" in text:
        return (
            f"Groq says the model is unavailable. The app is using '{MODEL_NAME}'. "
            "Check the model list in the Groq console and update MODEL_NAME in research_agent.py."
        )
    if "sqlite" in text:
        return (
            "The server's SQLite version is too old for CrewAI. Make sure "
            "'pysqlite3-binary' is in requirements.txt and that app.py was not changed."
        )
    if "connection" in text or "timeout" in text or "timed out" in text or "network" in text:
        return "Network problem: could not reach Groq or the web. Check your internet connection and try again."
    if isinstance(error, ImportError) and ("litellm" in text or "fallback" in text):
        # Only blame installation when the package genuinely failed to import.
        return (
            "CrewAI could not load its Groq connector (LiteLLM), so it is not installed "
            "correctly. Make sure requirements.txt contains 'crewai[litellm]' and reinstall. "
            f"Details: {type(error).__name__}: {error}"
        )
    if "tool_use_failed" in text or "failed to call a function" in text:
        return (
            "The model produced an invalid tool call (a known quirk of this model on Groq). "
            "Please click Start Research again. "
            f"Details: {str(error)[:600]}"
        )
    # Unknown error: show the REAL message so it can actually be diagnosed.
    return f"The research failed. Details: {type(error).__name__}: {str(error)[:1200]}"


# ---------------------------------------------------------------------------
# 4. Source checking
# ---------------------------------------------------------------------------
def find_unverified_urls(report, collected):
    """Return URLs that appear in the report but were NOT returned by the search tool."""
    real = {item["url"].rstrip("/") for item in collected}
    found = re.findall(r"https?://[^\s)\]>\"'<]+", report)
    bad = []
    for url in found:
        # Strip trailing punctuation and markdown symbols (e.g. **bold**, `code`).
        clean = url.rstrip(".,;:*_`").rstrip("/")
        if clean and clean not in real and clean not in bad:
            bad.append(clean)
    return bad


# ---------------------------------------------------------------------------
# 5. The research function the UI calls
# ---------------------------------------------------------------------------
def build_crew(llm, search_tool):
    """Create a fresh agent + task + crew (a fresh one is needed for a clean retry)."""
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

    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


def run_research(topic):
    topic = (topic or "").strip()
    if not topic:
        raise ResearchError("Please enter a research topic first.")
    if len(topic) > 300:
        raise ResearchError("Please keep the topic under 300 characters.")

    llm = get_llm()
    collected = []
    search_tool = make_search_tool(collected)
    inputs = {"topic": topic, "today": date.today().isoformat()}

    result = None
    for attempt in range(2):  # one automatic retry, only for Groq rate limits
        try:
            result = build_crew(llm, search_tool).kickoff(inputs=inputs)
            break
        except ResearchError:
            raise
        except Exception as e:
            traceback.print_exc()  # full details go to the terminal / Streamlit "Manage app" logs
            if attempt == 0 and is_rate_limit(e):
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue
            raise ResearchError(explain_error(e)) from e

    report = str(getattr(result, "raw", result) or "").strip()

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
