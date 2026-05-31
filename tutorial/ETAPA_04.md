# Etapa 4 — Infraestrutura: MLflow via Docker

## O que vamos aprender

- Docker Compose para subir o MLflow localmente
- Backend store (SQLite) vs Artifact store
- Mapeamento de portas (5050 → 5000)

---

## Passo 4.1 — Criar `docker/docker-compose.yml`

```yaml
services:
  mlflow-ui:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow_server
    platform: linux/arm64
    ports:
      - "5050:5000"
    volumes:
      - ./mlflow_data:/mlflow_data
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:////mlflow_data/mlflow.db
      --default-artifact-root mlflow-artifacts:/
      --serve-artifacts
      --artifacts-destination file:///mlflow_data/artifacts
```

> **Nota sobre `platform: linux/arm64`**: se você estiver em um Mac com chip Apple Silicon (M1/M2/M3/M4), esta linha garante que o Docker use a imagem correta. Em máquinas Intel/AMD, remova ou altere para `linux/amd64`.

---

## Pontos de aprendizado

### Arquitetura do MLflow Server

```
┌──────────────────────────────────────────┐
│          Docker Container                │
│                                          │
│   MLflow Server (:5000 interno)          │
│      │                                   │
│      ├── Backend Store (SQLite)          │
│      │   └── mlflow_data/mlflow.db       │
│      │       Armazena: runs, params,     │
│      │       tags, métricas              │
│      │                                   │
│      └── Artifact Store                  │
│          └── mlflow_data/artifacts       │
│              Armazena: inputs, outputs,  │
│              modelos                     │
│                                          │
└──────────────────────────────────────────┘
          ▲                    ▲
          │ HTTP API           │ Browser
     Seu código Python    http://localhost:5050
     (porta 5050)
```

### Mapeamento de portas

O container roda o MLflow na porta **5000** internamente, mas mapeamos para **5050** no host:
- `"5050:5000"` → `porta_host:porta_container`
- Seu código Python e o browser acessam via `http://localhost:5050`
- O `.env` deve ter `MLFLOW_TRACKING_URI=http://localhost:5050`

### Backend Store vs Artifact Store

- **Backend Store (SQLite)**: armazena metadados — runs, parâmetros, tags, métricas. Em produção seria PostgreSQL.
- **Artifact Store**: armazena arquivos — inputs, outputs, modelos. Em produção seria Azure Blob Storage / S3.
- **Volume `./mlflow_data`**: persiste dados entre restarts do container (mapeado para o diretório local `docker/mlflow_data`).

---

## Testando a infraestrutura

```bash
# 1. Subir o MLflow (a partir da raiz do projeto)
docker compose -f docker/docker-compose.yml up -d

# 2. Verificar que está rodando
curl http://localhost:5050/health
# Esperado: OK

# 3. Abrir no browser
# http://localhost:5050

# 4. Para derrubar
docker compose -f docker/docker-compose.yml down
```

---

## Próxima etapa

➡️ [Etapa 5 — Agente com Mock do LLM](ETAPA_05.md)
