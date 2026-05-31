# 🤖 HIAE AI Agent Template

> Reference template for generative AI agents — HIAE AI Platform

**🇧🇷 [Leia em Português](README_pt_Br.md)**

---

## Overview

This is a **starter template** for building, tracking, and evaluating generative AI agents on the HIAE AI Platform. It provides a production-ready structure with integrated **MLflow** tracking, **LLM-as-Judge** evaluation, **prompt versioning**, and a **mock agent** for local development without credentials.

## Architecture

```
agent-template/
├── agents/                  # Agent implementations
│   ├── agent.py             # Real agent (LLM via LiteLLM Proxy)
│   └── agent_mock.py        # Mock agent for local testing
├── config/                  # Configuration & infrastructure
│   ├── __init__.py          # Exports track_agent decorator
│   ├── app_config.py        # Environment-based configuration (dataclass)
│   ├── judge_config.py      # LLM Judge provider setup (LiteLLM Proxy patch)
│   └── tracking.py          # @track_agent decorator (MLflow autolog)
├── dataset/
│   └── register_dataset.py  # Evaluation dataset (inputs + expectations)
├── docker/
│   └── docker-compose.yml   # MLflow server (SQLite + local artifacts)
├── judge/
│   └── register_judge.py    # LLM-as-Judge evaluation pipeline
├── prompt/
│   ├── register_prompt_v1.py  # Prompt v1 registration in MLflow
│   └── register_prompt_v2.py  # Prompt v2 with stricter rules
├── .env_example             # Environment variables template
├── pyproject.toml           # Project metadata & dependencies
└── uv.lock                  # Dependency lock file (uv)
```

## Key Features

| Feature | Description |
|---|---|
| **MLflow Tracking** | Automatic experiment logging with the `@track_agent` decorator — inputs, outputs, traces, and governance tags |
| **Prompt Versioning** | Register and version prompts in MLflow Registry with `register_prompt_v1.py` and `register_prompt_v2.py` |
| **LLM-as-Judge** | Automated evaluation using Correctness, RelevanceToQuery, and custom Guidelines scorers via `register_judge.py` |
| **Mock Agent** | `agent_mock.py` allows full integration testing without LLM credentials |
| **LiteLLM Proxy** | Routes all LLM calls (agent + judges) through a centralized proxy |
| **Docker MLflow** | One-command MLflow server via `docker-compose.yml` |

## Prerequisites

- **Python** ≥ 3.12
- **[uv](https://docs.astral.sh/uv/)** (recommended package manager)
- **Docker** & **Docker Compose** (for MLflow server)

## Quick Start

### 1. Clone and configure

```bash
git clone <repository-url>
cd agent-template
cp .env_example .env
```

Edit `.env` with your values:

```env
AGENT_NAME=my-agent
TEAM_NAME=my-team
DOMAIN=my-domain
ENVIRONMENT=dev
CLIENT_ID=<your-client-id>
CLIENT_SECRET=<your-client-secret>
LLM_BASE_URL=<litellm-proxy-url>
LLM_API_KEY=<your-api-key>
MLFLOW_TRACKING_URI=http://localhost:5050
TAVILY_API_KEY=<optional-tavily-key>
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Start MLflow server

```bash
docker compose -f docker/docker-compose.yml up -d
```

MLflow UI will be available at **http://localhost:5050**.

### 4. Register a prompt

```bash
uv run python prompt/register_prompt_v1.py
```

### 5. Run the agent

**With real LLM:**
```bash
uv run python agents/agent.py "What is the capital of Brazil?"
```

**With mock (no credentials needed):**
```bash
uv run python agents/agent_mock.py "What is the capital of Brazil?"
```

### 6. Evaluate with LLM Judges

```bash
uv run python judge/register_judge.py
```

This will:
1. Load the evaluation dataset (4 test cases)
2. Run the agent for each query
3. Score responses using 3 LLM judges (Correctness, RelevanceToQuery, Guidelines)
4. Log all results to MLflow

## How It Works

### The `@track_agent` Decorator

The `@track_agent` decorator (defined in `config/tracking.py`) wraps your agent function and automatically handles all MLflow integration:

```
┌─────────────────────────────────────────────────────┐
│  BEFORE (injected by @track_agent)                  │
│  1. load_dotenv()                                   │
│  2. cfg = load_config()                             │
│  3. mlflow.set_tracking_uri(...)                    │
│  4. mlflow.set_experiment("/<team>/<domain>/<agent>")│
│  5. mlflow.langchain.autolog(log_traces=True)       │
│  6. mlflow.start_run("<agent>-<env>")               │
│  7. mlflow.set_tags({team, domain, ...})            │
│  8. mlflow.log_text(input, "input/...")             │
├─────────────────────────────────────────────────────┤
│  YOUR CODE (untouched)                              │
│  result = your_agent_logic(message)                 │
├─────────────────────────────────────────────────────┤
│  AFTER (injected by @track_agent)                   │
│  9. mlflow.log_text(result, "output/...")           │
│  10. print(run summary)                             │
│  11. mlflow.end_run()                               │
└─────────────────────────────────────────────────────┘
```

### Configuration

The `Config` dataclass (in `config/app_config.py`) loads all settings from environment variables:

| Variable | Description | Default |
|---|---|---|
| `AGENT_NAME` | Agent identifier (required) | — |
| `TEAM_NAME` | Team name for experiment hierarchy | `default` |
| `DOMAIN` | Business domain | `geral` |
| `ENVIRONMENT` | Deployment environment | `dev` |
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5050` |
| `LLM_BASE_URL` | LiteLLM Proxy URL | — |
| `LLM_API_KEY` | API key for LLM access (required) | — |

### Prompt Versioning

Prompts are registered in MLflow and loaded at runtime:

```python
# Register (run once)
mlflow.genai.register_prompt(name="my_prompt", template=[...])

# Load in agent
prompt = mlflow.genai.load_prompt("prompts:/my_prompt@latest")
messages = prompt.format(domain=domain, user_message=message)
```

Supported references:
- `prompts:/my_prompt@latest` — latest version
- `prompts:/my_prompt/1` — specific version
- `prompts:/my_prompt@prod` — alias (e.g., production)

### LLM Judges

The evaluation pipeline in `judge/register_judge.py` uses three scorers:

| Judge | Purpose |
|---|---|
| **Correctness** | Verifies the response contains expected facts |
| **RelevanceToQuery** | Checks if the response is relevant to the question |
| **Guidelines** | Validates custom rules (language, length, disclaimers) |

The `judge_config.py` patches MLflow's OpenAI provider to route judge calls through the LiteLLM Proxy instead of `api.openai.com`.

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Type check
uv run mypy .
```

## Tech Stack

- **Python** 3.12+
- **LangChain** / **LangGraph** — Agent framework
- **MLflow** 3.x — Experiment tracking, prompt registry, evaluation
- **LiteLLM Proxy** — Unified LLM gateway
- **Docker** — Local MLflow server
- **uv** — Fast Python package manager

---

## License

This project is internal to HIAE. See your organization's policies for usage terms.
