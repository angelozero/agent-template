# Etapa 4 — Infraestrutura: MLflow via Docker

## O que vamos aprender

- Docker Compose para subir o MLflow localmente
- Backend store (SQLite) vs Artifact store
- Healthcheck

---

## Passo 4.1 — Criar `docker-compose.yaml`

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v3.12.0
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    command:
      - mlflow
      - server
      - --host
      - "0.0.0.0"
      - --port
      - "5000"
      - --backend-store-uri
      - sqlite:////mlflow/mlflow.db       # metadados (runs, params, tags)
      - --artifacts-destination
      - /mlflow/artifacts                  # arquivos (inputs, outputs)
      - --workers
      - "2"
    mem_limit: 2g
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s

volumes:
  mlflow-data:
    driver: local
```

---

## Pontos de aprendizado

### Arquitetura do MLflow Server

```
┌──────────────────────────────────────────┐
│          Docker Container                │
│                                          │
│   MLflow Server (:5000)                  │
│      │                                   │
│      ├── Backend Store (SQLite)          │
│      │   └── mlflow.db                   │
│      │       Armazena: runs, params,     │
│      │       tags, métricas              │
│      │                                   │
│      └── Artifact Store                  │
│          └── /mlflow/artifacts           │
│              Armazena: inputs, outputs,  │
│              modelos                     │
│                                          │
└──────────────────────────────────────────┘
         ▲                    ▲
         │ HTTP API           │ Browser
    Seu código Python    http://localhost:5000
```

- **Backend Store (SQLite)**: armazena metadados — runs, parâmetros, tags, métricas. Em produção seria PostgreSQL.
- **Artifact Store**: armazena arquivos — inputs, outputs, modelos. Em produção seria Azure Blob Storage / S3.
- **Volume `mlflow-data`**: persiste dados entre restarts do container.

---

## Passo 4.2 — Criar o `Justfile`

```just
default:
    @just --list

dc := if os() == "macos" { "docker-compose" } else { "docker compose" }

# Instala dependências com uv
install:
    uv sync --all-groups

# Sobe o MLflow GenAI local (http://localhost:5000)
up:
    {{dc}} up -d
    @echo ""
    @echo "  MLflow UI → http://localhost:5000"
    @echo ""

# Derruba o MLflow
down:
    {{dc}} down

# Exibe logs do MLflow em tempo real
logs:
    {{dc}} logs -f mlflow

# Executa o agente diretamente
run agent_file message:
    uv run python "{{agent_file}}" "{{message}}"

# Roda testes
test:
    uv run pytest

# Verifica código
lint:
    uv run ruff check .

# Formata código
format:
    uv run ruff format .
```

---

## Testando a infraestrutura

```bash
# 1. Subir o MLflow
just up

# 2. Verificar que está rodando
curl http://localhost:5000/health
# Esperado: OK

# 3. Abrir no browser
# http://localhost:5000
```

---

## Próxima etapa

➡️ [Etapa 5 — Agente com Mock do LLM](ETAPA_05.md)
