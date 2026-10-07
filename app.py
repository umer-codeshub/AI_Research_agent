"""
app.py - the Streamlit user interface.

This file only handles input and display. All AI logic is in research_agent.py.
Run locally with:  streamlit run app.py
"""

import os

import streamlit as st

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

# Streamlit Cloud stores secrets in st.secrets. Our agent reads os.environ,
# so we copy the key across. Locally there is no secrets file, so we ignore the error.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"])
except Exception:
    pass

# Import the agent. If a package is missing/broken, show a clear message instead of a traceback.
try:
    from research_agent import ResearchError, run_research
except ImportError as import_error:
    st.error(
        "A required package could not be imported. "
        "Check that requirements.txt installed correctly and that you are using Python 3.10-3.13."
    )
    st.code(str(import_error))
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
)
start = st.button("Start Research", type="primary")

# --- Run the research ------------------------------------------------------
if start:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.pop("result", None)
        try:
            with st.spinner("Searching the web and writing your report... (can take 30-90 seconds)"):
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
            st.markdown(f"{i}. [{src['title'] or src['url']}]({src['url']})")

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
