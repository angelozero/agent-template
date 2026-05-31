# Etapa 2 — Configuração com Variáveis de Ambiente

## O que vamos aprender

- Padrão de configuração via `.env` + dataclass
- Validação de variáveis obrigatórias
- Separação de responsabilidades (config isolada da lógica)

---

## Passo 2.1 — Criar o `.env_example`

```env
AGENT_NAME=
TEAM_NAME=
DOMAIN=
ENVIRONMENT=
CLIENT_ID=
CLIENT_SECRET=
LLM_BASE_URL=
LLM_API_KEY=
MLFLOW_TRACKING_URI=
TAVILY_API_KEY=
```

> **Sobre a `LLM_BASE_URL`**: esta é a URL do **endpoint do LLM Gateway** — o endereço para onde as chamadas de chat/completion são enviadas. **Não é** uma URL de autenticação. O LangChain vai fazer `POST {base_url}/chat/completions`.

---

## Passo 2.2 — Criar `config/app_config.py`

```python
import os
from dataclasses import dataclass


# @dataclasse ---> Cria um objeto imutável que ninguém pode alterar.
@dataclass(frozen=True)
class Config:
    agent_name: str
    team: str
    domain: str
    enviroment: str
    mlflow_tracking_uri: str
    llm_base_url: str
    llm_api_key: str

def load_config() -> Config:
    agent_name = os.environ.get("AGENT_NAME")
    if not agent_name:
        raise EnvironmentError("AGENT_NAME value was not found in .env file")

    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key or llm_api_key == "define":
        raise EnvironmentError("LLM_API_KEY value was not found in .env file")

    return Config(
        agent_name=agent_name,
        team=os.getenv("TEAM_NAME", "default"),
        domain=os.getenv("DOMAIN", "geral"),
        enviroment=os.getenv("ENVIROMENT", "dev"),
        mlflow_tracking_uri=os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5050"
        ),
        llm_base_url=os.getenv(
            "LLM_BASE_URL", "YOUR_LLM_URL"
        ),
        llm_api_key=llm_api_key,
    )
```

## Passo 2.3 — Criar `config/__init__.py`

```python
from config.tracking import track_agent

__all__ = ["track_agent"]
```

Este arquivo expõe o decorator `track_agent` para que outros módulos possam importar diretamente com `from config import track_agent`.

---

## Pontos de aprendizado

### `@dataclass(frozen=True)`

Cria um objeto **imutável**. Uma vez criada a config, ninguém pode alterar. Isso é uma boa prática para configurações — equivalente a um `record` do Java 17+:

```java
// Java equivalente:
public record Config(
    String agentName,
    String team,
    String domain,
    String enviroment,
    String mlflowTrackingUri,
    String llmBaseUrl,
    String llmApiKey
) {}
```

### Validação fail-fast

Se `AGENT_NAME` ou `LLM_API_KEY` não estiverem definidos, o programa falha **imediatamente** com mensagem clara — em vez de falhar misteriosamente depois.

### Defaults sensatos

`TEAM_NAME` tem default `"default"`, `ENVIROMENT` tem default `"dev"`. Isso permite rodar rapidamente sem configurar tudo.

### Diferença entre `==` e `is`

No código usamos `llm_api_key == "define"` (comparação de valor). Em Python:
- `==` compara se os **valores** são iguais
- `is` compara se são o **mesmo objeto** na memória

Para strings e valores, use sempre `==`.

---

## Próxima etapa

➡️ [Etapa 3 — O Coração: Integração com MLflow via @track_agent](ETAPA_03.md)
