# Etapa 2 — Configuração com Variáveis de Ambiente

## O que vamos aprender

- Padrão de configuração via `.env` + dataclass
- Validação de variáveis obrigatórias
- Separação de responsabilidades (config isolada da lógica)

---

## Passo 2.1 — Criar o `.env.example`

```env
# ─── Identidade do agente ────────────────────────────────────────────────────
AGENT_NAME=meu-agente
TEAM_NAME=meu-time
DOMAIN=geral
ENVIRONMENT=dev

# ─── LLM Gateway ─────────────────────────────────────────────────────────────
LLM_BASE_URL=https://flow.ciandt.com/flow-llm-proxy/v1
LLM_API_KEY=define

# ─── MLflow local ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI=http://localhost:5000
```

> **Sobre a `LLM_BASE_URL`**: esta é a URL do **endpoint do LLM Gateway** — o endereço para onde as chamadas de chat/completion são enviadas. **Não é** uma URL de autenticação. O LangChain vai fazer `POST {base_url}/chat/completions`.

---

## Passo 2.2 — Criar `ai_platform/config.py`

```python
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    agent_name: str
    team_name: str
    domain: str
    environment: str
    mlflow_tracking_uri: str
    llm_base_url: str
    llm_api_key: str


def load_config() -> Config:
    agent_name = os.environ.get("AGENT_NAME")
    if not agent_name:
        raise EnvironmentError("AGENT_NAME não definido. Configure no arquivo .env")

    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key or llm_api_key == "define":
        raise EnvironmentError(
            "LLM_API_KEY não definido. Configure com o token no arquivo .env"
        )

    return Config(
        agent_name=agent_name,
        team_name=os.getenv("TEAM_NAME", "default"),
        domain=os.getenv("DOMAIN", "geral"),
        environment=os.getenv("ENVIRONMENT", "dev"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://flow.ciandt.com/flow-llm-proxy/v1"),
        llm_api_key=llm_api_key,
    )
```

---

## Pontos de aprendizado

### `@dataclass(frozen=True)`

Cria um objeto **imutável**. Uma vez criada a config, ninguém pode alterar. Isso é uma boa prática para configurações — equivalente a um `record` do Java 17+:

```java
// Java equivalente:
public record Config(
    String agentName,
    String teamName,
    String domain,
    String environment,
    String mlflowTrackingUri,
    String llmBaseUrl,
    String llmApiKey
) {}
```

### Validação fail-fast

Se `AGENT_NAME` ou `LLM_API_KEY` não estiverem definidos, o programa falha **imediatamente** com mensagem clara — em vez de falhar misteriosamente depois.

### Defaults sensatos

`TEAM_NAME` tem default `"default"`, `ENVIRONMENT` tem default `"dev"`. Isso permite rodar rapidamente sem configurar tudo.

---

## Próxima etapa

➡️ [Etapa 3 — O Coração: Integração com MLflow via @track_agent](ETAPA_03.md)
