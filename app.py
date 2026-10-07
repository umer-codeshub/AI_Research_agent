"""
app.py - the Streamlit user interface.

This file only handles input and display. All AI logic is in research_agent.py.
Run locally with:  streamlit run app.py
"""

# --- Must run BEFORE anything imports sqlite3 / crewai / chromadb ----------
# Streamlit Cloud's system SQLite is too old for ChromaDB (used by CrewAI).
# If pysqlite3 is installed (Linux), use it instead. Otherwise do nothing.
import sys

try:
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import os

# Turn off CrewAI telemetry: it tries to register signal handlers and make
# network calls, which can misbehave inside Streamlit's script thread.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import streamlit as st

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

# Streamlit Cloud stores secrets in st.secrets. Our agent reads os.environ,
# so we copy the key across. Locally there is no secrets file, so we ignore the error.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"]).strip().strip("\"'")
except Exception:
    pass

# Import the agent. Catch ANY import-time failure (not only ImportError), because
# version conflicts often raise RuntimeError / AttributeError / pydantic errors.
try:
    from research_agent import ResearchError, run_research
except Exception as import_error:
    st.error(
        "The research agent could not be loaded. Check that requirements.txt installed "
        "correctly and that you are using Python 3.10-3.13 (3.12 recommended)."
    )
    st.code(f"{type(import_error).__name__}: {import_error}")
    st.stop()

# --- Page header -----------------------------------------------------------
st.title("🔎 AI Research Agent")
st.write(
    "Enter a topic. The agent searches the web (DuckDuckGo), reads the results, "
    "and writes a structured report. Claims it cannot verify are labeled as such."
)

topic = st.text_input(
    "Enter a topic to research:",
    placeholder="e.g. How do solid-state batteries work?",
    max_chars=300,
)
start = st.button("Start Research", type="primary")

# --- Run the research ------------------------------------------------------
if start:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.pop("result", None)
        try:
            with st.spinner(
                "Searching the web and writing your report... (can take 30-90 seconds, "
                "please don't click anything else)"
            ):
                st.session_state["result"] = run_research(topic)
                st.session_state["topic"] = topic.strip()
        except ResearchError as e:
            st.error(str(e))
        except Exception as e:  # last safety net
            st.error(f"Something unexpected went wrong: {type(e).__name__}: {e}")

# --- Show the result (kept in session_state so it survives button clicks) ---
result = st.session_state.get("result")
if result:
    st.divider()
    st.markdown(result["report"])

    if result["unverified_urls"]:
        st.warning(
            "These URLs appear in the report but were NOT returned by the search tool. "
            "Treat them as unverified and do not trust them:\n\n"
            + "\n".join(f"- {u}" for u in result["unverified_urls"])
        )

    with st.expander(f"Search results the agent actually retrieved ({len(result['sources'])})"):
        for i, src in enumerate(result["sources"], start=1):
            label = (src["title"] or src["url"]).replace("[", "(").replace("]", ")")
            st.markdown(f"{i}. [{label}]({src['url']})")

    st.download_button(
        "Download report (.md)",
        data=result["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
    st.caption(
        "AI-generated from short web search snippets. Always open the sources and "
        "verify important claims yourself."
    )
