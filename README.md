# AI Research Agent

A beginner-friendly, single-agent research app. You enter a topic; a CrewAI agent searches the web with DuckDuckGo, reads the results, and writes a structured report using Groq's `openai/gpt-oss-120b` model. The report is shown in a Streamlit app.

This is a learning project. It is **not** a replacement for real research.

## Features

- One CrewAI agent with one custom web search tool (no search API key needed)
- Structured report: summary, findings, analysis, evidence, perspectives, limitations, conclusion, sources
- Anti-hallucination rules in the agent's instructions (no invented sources, URLs or statistics)
- Automatic check: URLs in the report that the search tool never returned are flagged as unverified
- Refuses to show a report if the web search returned nothing
- Friendly error messages (missing/invalid key, rate limits, network problems)
- Report download as Markdown

## Tech stack

| Part | Choice |
|---|---|
| Language | Python 3.10 - 3.13 (3.12 recommended) |
| Agent framework | CrewAI (`crewai[litellm]`) |
| LLM provider | Groq, model `openai/gpt-oss-120b` (used as `groq/openai/gpt-oss-120b` in CrewAI) |
| Web search | `ddgs` (DuckDuckGo, no API key) |
| UI | Streamlit |
| Hosting | Streamlit Community Cloud |

## Project structure

```text
ai-research-agent/
├── app.py              # Streamlit UI (input and display only)
├── research_agent.py   # Search tool, Groq LLM, CrewAI agent/task/crew, error handling
├── requirements.txt    # Python dependencies
├── .env.example        # Template for your API key (safe to commit)
├── .gitignore          # Keeps .env and secrets out of Git
├── README.md           # This file
└── assets/             # Screenshots for the README
```

## How it works

```text
User -> Streamlit UI -> run_research() -> CrewAI Agent <-> web_search tool (DuckDuckGo)
                                              |
                                          Groq LLM
                                              |
                                        Research report -> Streamlit UI
```

## Installation

1. Install Python 3.12 (3.10-3.13 work; **3.14 does not**).
2. Open a terminal in the project folder, then create and activate a virtual environment:

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1
# Windows Command Prompt
venv\Scripts\activate.bat
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment variables

1. Create a free Groq API key at https://console.groq.com/keys
2. Copy `.env.example` to `.env`:

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

3. Open `.env` and replace the placeholder:

```text
GROQ_API_KEY=gsk_your_real_key_here
```

No quotes, no spaces. **Never commit `.env`.**

## Run locally

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501.

## How to use

1. Type a topic or question.
2. Click **Start Research**. It usually takes 30-90 seconds.
3. Read the report. Open the "Search results the agent actually retrieved" box to see real URLs.
4. Check any warning about unverified URLs, and verify important claims yourself.

## Push to GitHub

```bash
git init
git add .
git status          # make sure .env is NOT listed
git commit -m "Initial commit: AI research agent"
git branch -M main
```

Create an empty repository on https://github.com/new (no README, no .gitignore, since we already have them), then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/ai-research-agent.git
git push -u origin main
```

Upload: all files in the structure above. Do **not** upload: `.env`, `venv/`, `__pycache__/`, `.streamlit/secrets.toml`.

If you ever commit a real key by mistake, delete that key in the Groq console immediately and create a new one. Deleting the file from Git is not enough because it stays in the history.

## Deploy to Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** and choose to deploy from an existing repo.
3. Select your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings**:
   - Choose Python **3.12** (if there is a version choice).
   - In **Secrets**, paste exactly (TOML format, with quotes):

```toml
GROQ_API_KEY = "gsk_your_real_key_here"
```

5. Click **Deploy**. The first build takes a few minutes.
6. If it fails, open **Manage app** (bottom right) and read the logs.

You can edit secrets later under **App settings -> Secrets**.

## Limitations

- The agent only sees short search snippets, not full web pages. Details can be missing.
- DuckDuckGo search is unofficial and free. It can rate-limit you, return poor results, or block some cloud IPs (this can happen on Streamlit Cloud).
- The model can still make mistakes or misread snippets, even with strict instructions. The app reduces hallucination; it cannot eliminate it.
- The URL check only proves a URL was returned by search. It does not prove the page says what the report claims.
- Groq's free tier has rate limits.
- No memory, no history, no login. Each run is independent.

## Future improvements

- Fetch and read full pages for the top results
- Show search queries the agent used
- Save past reports
- Add a news-focused search mode

## Troubleshooting

| Problem | Meaning and fix |
|---|---|
| `ModuleNotFoundError: No module named 'crewai'` | Packages are not installed in the active environment. Activate the venv and run `pip install -r requirements.txt`. |
| `ImportError` for `crewai.tools` or `BaseTool` | Old CrewAI version. Run `pip install -U "crewai[litellm]"`. |
| Error mentioning `litellm` | The Groq connector is missing. Make sure `requirements.txt` contains `crewai[litellm]`, not plain `crewai`. |
| `GROQ_API_KEY is missing` | `.env` does not exist, is misnamed (e.g. `.env.txt`), or is in the wrong folder. It must sit next to `app.py`. On Streamlit Cloud, add the secret. |
| 401 / authentication error | The key is wrong, has quotes or spaces, or was deleted. Create a new one. |
| Model not found | Groq renamed or retired the model. Check the Groq console model list and change `MODEL_NAME` in `research_agent.py`. |
| 429 / rate limit | Too many requests. Wait a minute. |
| "web search returned no results" | DuckDuckGo throttled or blocked you. Wait and retry, or try a different network. |
| Install fails on Python 3.14 | CrewAI requires Python below 3.14. Use 3.12. |
| Streamlit Cloud build fails | Read the logs. Common causes: wrong Python version, a typo in `requirements.txt`, or a dependency conflict. |
