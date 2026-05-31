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
3. **Iterar rapidamente** no desenvolvimento do SDK

Quando quiser usar o LLM real, basta substituir `MockLLM` por `init_chat_model()` do LangChain (próxima etapa).

---

## Passo 5.1 — Criar `examples/agent.py` com mock

```python
"""Agente com LLM mockado para testar integração MLflow."""
import os
from dotenv import load_dotenv
from ai_platform import track_agent

load_dotenv()


class MockLLM:
    """Simula um LLM que responde com texto fixo."""

    def invoke(self, messages):
        user_msg = messages[-1]["content"] if isinstance(messages[-1], dict) else str(messages[-1])
        return {
            "role": "assistant",
            "content": f"[MOCK] Recebi sua mensagem: '{user_msg}'. "
                       f"Esta é uma resposta simulada do LLM.",
            "model": "mock-gpt-4.1",
            "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35},
        }


def build_agent():
    """Constrói o 'agente' — neste caso, apenas o mock."""
    return MockLLM()


@track_agent
def invoke_agent(message: str):
    """Ponto de entrada do agente. Mantenha esta assinatura."""
    agent = build_agent()
    result = agent.invoke([{"role": "user", "content": message}])
    return result


if __name__ == "__main__":
    import sys
    message = sys.argv[1] if len(sys.argv) > 1 else "Qual a previsão do tempo em São Paulo?"
    invoke_agent(message)
```

---

## Passo 5.2 — Configurar o `.env` para o mock

```bash
cp .env.example .env
```

Edite o `.env`:

```env
AGENT_NAME=tutorial-agente
TEAM_NAME=meu-time
DOMAIN=geral
ENVIRONMENT=dev
LLM_API_KEY=mock-key-123
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## Passo 5.3 — Executar e verificar

```bash
# 1. Certifique-se que o MLflow está rodando
just up

# 2. Instale as dependências
just install

# 3. Execute o agente mockado
python examples/agent.py "Olá, mundo!"

# 4. Saída esperada:
# ────────────────────────────────────────────────────
#   MLflow Run ID  : abc123...
#   Experimento    : /meu-time/geral/tutorial-agente
#   UI             : http://localhost:5000
# ────────────────────────────────────────────────────
```

---

## O que você verá no MLflow (http://localhost:5000)

```
Experiments
└── /meu-time/geral/tutorial-agente
    └── Run: tutorial-agente-dev
        ├── Tags:
        │   ├── ai_platform.agent_name = tutorial-agente
        │   ├── ai_platform.team = meu-time
        │   ├── ai_platform.domain = geral
        │   └── ai_platform.environment = dev
        └── Artifacts:
            ├── input/user_message.txt → "Olá, mundo!"
            └── output/final_response.json → {role: assistant, ...}
```

> **Nota:** Com o mock, não haverá traces de LLM (pois o mock não passa pelo LangChain). Os traces aparecerão quando usarmos o LLM real na próxima etapa.

---

## Próxima etapa

➡️ [Etapa 6 — Substituir o Mock por LLM Real](ETAPA_06.md)
