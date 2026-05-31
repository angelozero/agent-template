# 🤖 HIAE AI Agent Template

> Template de referência para agentes de IA generativa — HIAE AI Platform

**🇺🇸 [Read in English](README.md)**

---

## Visão Geral

Este é um **template inicial** para construir, rastrear e avaliar agentes de IA generativa na Plataforma de IA HIAE. Ele fornece uma estrutura pronta para produção com rastreamento integrado via **MLflow**, avaliação **LLM-as-Judge**, **versionamento de prompts** e um **agente mock** para desenvolvimento local sem credenciais.

## Arquitetura

```
agent-template/
├── agents/                  # Implementações do agente
│   ├── agent.py             # Agente real (LLM via LiteLLM Proxy)
│   └── agent_mock.py        # Agente mock para testes locais
├── config/                  # Configuração e infraestrutura
│   ├── __init__.py          # Exporta o decorator track_agent
│   ├── app_config.py        # Configuração via variáveis de ambiente (dataclass)
│   ├── judge_config.py      # Setup do provider dos LLM Judges (patch LiteLLM Proxy)
│   └── tracking.py          # Decorator @track_agent (MLflow autolog)
├── dataset/
│   └── register_dataset.py  # Dataset de avaliação (inputs + expectativas)
├── docker/
│   └── docker-compose.yml   # Servidor MLflow (SQLite + artefatos locais)
├── judge/
│   └── register_judge.py    # Pipeline de avaliação LLM-as-Judge
├── prompt/
│   ├── register_prompt_v1.py  # Registro do prompt v1 no MLflow
│   └── register_prompt_v2.py  # Prompt v2 com regras mais rígidas
├── .env_example             # Template de variáveis de ambiente
├── pyproject.toml           # Metadados e dependências do projeto
└── uv.lock                  # Lock de dependências (uv)
```

## Funcionalidades Principais

| Funcionalidade | Descrição |
|---|---|
| **Rastreamento MLflow** | Log automático de experimentos com o decorator `@track_agent` — inputs, outputs, traces e tags de governança |
| **Versionamento de Prompts** | Registre e versione prompts no MLflow Registry com `register_prompt_v1.py` e `register_prompt_v2.py` |
| **LLM-as-Judge** | Avaliação automatizada usando scorers de Correctness, RelevanceToQuery e Guidelines customizadas via `register_judge.py` |
| **Agente Mock** | `agent_mock.py` permite testes de integração completos sem credenciais de LLM |
| **LiteLLM Proxy** | Roteia todas as chamadas LLM (agente + judges) por um proxy centralizado |
| **Docker MLflow** | Servidor MLflow com um único comando via `docker-compose.yml` |

## Pré-requisitos

- **Python** ≥ 3.12
- **[uv](https://docs.astral.sh/uv/)** (gerenciador de pacotes recomendado)
- **Docker** & **Docker Compose** (para o servidor MLflow)

## Início Rápido

### 1. Clone e configure

```bash
git clone <url-do-repositorio>
cd agent-template
cp .env_example .env
```

Edite o `.env` com seus valores:

```env
AGENT_NAME=meu-agente
TEAM_NAME=meu-time
DOMAIN=meu-dominio
ENVIRONMENT=dev
CLIENT_ID=<seu-client-id>
CLIENT_SECRET=<seu-client-secret>
LLM_BASE_URL=<url-do-litellm-proxy>
LLM_API_KEY=<sua-api-key>
MLFLOW_TRACKING_URI=http://localhost:5050
TAVILY_API_KEY=<chave-tavily-opcional>
```

### 2. Instale as dependências

```bash
uv sync
```

### 3. Inicie o servidor MLflow

```bash
docker compose -f docker/docker-compose.yml up -d
```

A interface do MLflow estará disponível em **http://localhost:5050**.

### 4. Registre um prompt

```bash
uv run python prompt/register_prompt_v1.py
```

### 5. Execute o agente

**Com LLM real:**
```bash
uv run python agents/agent.py "Qual a capital do Brasil?"
```

**Com mock (sem credenciais):**
```bash
uv run python agents/agent_mock.py "Qual a capital do Brasil?"
```

### 6. Avalie com LLM Judges

```bash
uv run python judge/register_judge.py
```

Isso irá:
1. Carregar o dataset de avaliação (4 casos de teste)
2. Executar o agente para cada pergunta
3. Avaliar as respostas usando 3 LLM judges (Correctness, RelevanceToQuery, Guidelines)
4. Registrar todos os resultados no MLflow

## Como Funciona

### O Decorator `@track_agent`

O decorator `@track_agent` (definido em `config/tracking.py`) envolve a função do agente e gerencia automaticamente toda a integração com o MLflow:

```
┌─────────────────────────────────────────────────────┐
│  ANTES (injetado pelo @track_agent)                 │
│  1. load_dotenv()                                   │
│  2. cfg = load_config()                             │
│  3. mlflow.set_tracking_uri(...)                    │
│  4. mlflow.set_experiment("/<time>/<dominio>/<agente>")│
│  5. mlflow.langchain.autolog(log_traces=True)       │
│  6. mlflow.start_run("<agente>-<env>")              │
│  7. mlflow.set_tags({team, domain, ...})            │
│  8. mlflow.log_text(input, "input/...")             │
├─────────────────────────────────────────────────────┤
│  SEU CÓDIGO (intocado)                              │
│  result = sua_logica_do_agente(message)             │
├─────────────────────────────────────────────────────┤
│  DEPOIS (injetado pelo @track_agent)                │
│  9. mlflow.log_text(result, "output/...")           │
│  10. print(resumo da run)                           │
│  11. mlflow.end_run()                               │
└─────────────────────────────────────────────────────┘
```

### Configuração

O dataclass `Config` (em `config/app_config.py`) carrega todas as configurações a partir de variáveis de ambiente:

| Variável | Descrição | Padrão |
|---|---|---|
| `AGENT_NAME` | Identificador do agente (obrigatório) | — |
| `TEAM_NAME` | Nome do time para hierarquia de experimentos | `default` |
| `DOMAIN` | Domínio de negócio | `geral` |
| `ENVIRONMENT` | Ambiente de deploy | `dev` |
| `MLFLOW_TRACKING_URI` | URL do servidor MLflow | `http://localhost:5050` |
| `LLM_BASE_URL` | URL do LiteLLM Proxy | — |
| `LLM_API_KEY` | Chave de API para acesso ao LLM (obrigatório) | — |

### Versionamento de Prompts

Os prompts são registrados no MLflow e carregados em tempo de execução:

```python
# Registrar (executar uma vez)
mlflow.genai.register_prompt(name="meu_prompt", template=[...])

# Carregar no agente
prompt = mlflow.genai.load_prompt("prompts:/meu_prompt@latest")
messages = prompt.format(domain=domain, user_message=message)
```

Referências suportadas:
- `prompts:/meu_prompt@latest` — última versão
- `prompts:/meu_prompt/1` — versão específica
- `prompts:/meu_prompt@prod` — alias (ex: produção)

### LLM Judges

O pipeline de avaliação em `judge/register_judge.py` usa três scorers:

| Judge | Propósito |
|---|---|
| **Correctness** | Verifica se a resposta contém os fatos esperados |
| **RelevanceToQuery** | Verifica se a resposta é relevante à pergunta |
| **Guidelines** | Valida regras customizadas (idioma, tamanho, disclaimers) |

O `judge_config.py` aplica um patch no provider OpenAI do MLflow para rotear as chamadas dos judges pelo LiteLLM Proxy em vez de `api.openai.com`.

## Desenvolvimento

```bash
# Instalar dependências de desenvolvimento
uv sync --group dev

# Executar testes
uv run pytest

# Lint
uv run ruff check .

# Verificação de tipos
uv run mypy .
```

## Stack Tecnológica

- **Python** 3.12+
- **LangChain** / **LangGraph** — Framework de agentes
- **MLflow** 3.x — Rastreamento de experimentos, registro de prompts, avaliação
- **LiteLLM Proxy** — Gateway unificado de LLMs
- **Docker** — Servidor MLflow local
- **uv** — Gerenciador de pacotes Python rápido

---

## Licença

Este projeto é interno ao HIAE. Consulte as políticas da sua organização para termos de uso.
