# Etapa 5 — Agente com Mock do LLM

## O que vamos aprender

- Como criar um agente simples
- Como usar `@track_agent` na prática
- Como mockar o LLM Gateway para testar localmente sem credenciais

---

## Por que mockar?

O mock permite:

1. **Testar a integração MLflow** sem precisar de token/credenciais
2. **Verificar que experiments, runs, tags e artifacts** são criados corretamente
3. **Iterar rapidamente** no desenvolvimento

Quando quiser usar o LLM real, basta usar `agents/agent.py` com `init_chat_model()` do LangChain (próxima etapa).

---

## Passo 5.1 — Criar `agents/agent_mock.py`

```python
import os
from dotenv import load_dotenv
from config import track_agent

load_dotenv()

### Mock LLM
# Em vez de chamar o LLM Gateway real, retornamos respostas fixas.
# Isso permite testar toda a integração MLflow sem credenciais.


class MockLLM:
    """Simula um LLM que responde com texto fixo."""

    def invoke(self, messages):
        user_msg = (
            messages[-1]["content"]
            if isinstance(messages[-1], dict)
            else str(messages[-1])
        )
        return {
            "role": "assistant",
            "content": f"[MOCK] Recebi sua mensagem: '{user_msg}'. "
            f"Esta é uma resposta simulada do LLM.",
            "model": "mock-gpt-4.1",
            "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35},
        }


def build_model():
    """Constrói o 'agente' — neste caso, apenas o mock."""
    return MockLLM()


@track_agent
def invoke_agent(message: str):
    """Ponto de entrada do agente. Mantenha essa assinatura"""
    agent = build_model()
    result = agent.invoke([{"role": "user", "content": message}])
    return result


if __name__ == "__main__":
    import sys

    message = (
        sys.argv[1] if len(sys.argv) > 1 else "Qual a previsão do tempo em São Paulo?"
    )
    invoke_agent(message)
```

---

## Passo 5.2 — Configurar o `.env` para o mock

```bash
cp .env_example .env
```

Edite o `.env`:

```env
AGENT_NAME=tutorial-agente
TEAM_NAME=meu-time
DOMAIN=geral
ENVIROMENT=dev
LLM_API_KEY=mock-key-123
MLFLOW_TRACKING_URI=http://localhost:5050
```

---

## Passo 5.3 — Executar e verificar

```bash
# 1. Certifique-se que o MLflow está rodando
docker compose -f docker/docker-compose.yml up -d

# 2. Instale as dependências
uv sync --all-groups

# 3. Execute o agente mockado
uv run python agents/agent_mock.py "Olá, mundo!"

# 4. Saída esperada:
# ────────────────────────────────────────────────────
#   MLflow Run ID  : abc123...
#   Experimento    : /meu-time/geral/tutorial-agente
#   UI             : http://localhost:5050
# ────────────────────────────────────────────────────
```

---

## O que você verá no MLflow (http://localhost:5050)

```
Experiments
└── /meu-time/geral/tutorial-agente
    └── Run: tutorial-agente-dev
        ├── Tags:
        │   ├── ai_platform.agent_name = tutorial-agente
        │   ├── ai_platform.domain = geral
        │   ├── ai_platform.enviroment = dev
        │   ├── ai_platform.framework = langchain
        │   └── ai_platform.is_agent = true
        └── Artifacts:
            ├── input/user_message.txt → "Olá, mundo!"
            └── output/final_response.json → {role: assistant, ...}
```

> **Nota:** Com o mock, não haverá traces de LLM (pois o mock não passa pelo LangChain). Os traces aparecerão quando usarmos o LLM real na próxima etapa.

---

## Como o `@track_agent` funciona na prática

O decorator no `agents/agent_mock.py` faz o seguinte fluxo:

```
┌─────────────────────────────────────────────────────┐
│  ANTES (código injetado pelo @track_agent)          │
│                                                     │
│  1. load_dotenv()                                   │
│  2. cfg = load_config()                             │
│  3. mlflow.set_tracking_uri("http://localhost:5050")│
│  4. mlflow.set_experiment("/time/dominio/agente")   │
│  5. mlflow.langchain.autolog(log_traces=True)       │
│  6. mlflow.start_run("meu-agente-dev")              │
│  7. mlflow.set_tags({agent_name, domain, ...})      │
│  8. mlflow.log_text("Olá", "input/...")             │
├─────────────────────────────────────────────────────┤
│  EXECUÇÃO ORIGINAL (seu código, intocado)           │
│                                                     │
│  result = agent.invoke([...])                       │
│  return result                                      │
├─────────────────────────────────────────────────────┤
│  DEPOIS (código injetado pelo @track_agent)         │
│                                                     │
│  9. mlflow.log_dict(result, "output/...")            │
│  10. print("Run ID: abc123...")                      │
│  11. mlflow.end_run()                                │
└─────────────────────────────────────────────────────┘
```

---

## Próxima etapa

➡️ [Etapa 6 — Substituir o Mock por LLM Real](ETAPA_06.md)
