# Etapa 6 — Substituir o Mock por LLM Real via LiteLLM Proxy

## O que vamos aprender

- Como usar `init_chat_model` do LangChain com um LiteLLM Proxy
- Como configurar `base_url` e `api_key`
- Diferença entre traces com mock vs LLM real

---

## Passo 6.1 — Atualizar o `.env`

```env
AGENT_NAME=tutorial-agente
TEAM_NAME=meu-time
DOMAIN=geral
ENVIRONMENT=dev

# LLM Gateway (LiteLLM Proxy)
LLM_BASE_URL=https://flow.ciandt.com/flow-llm-proxy/v1
LLM_API_KEY=sua-api-key-aqui

MLFLOW_TRACKING_URI=http://localhost:5000
```

> **Atenção ao `/v1`**: o LangChain com `model_provider="openai"` espera que a URL termine em `/v1`, pois ele concatena `/chat/completions` no final.

---

## Passo 6.2 — Descobrir modelos disponíveis

Antes de rodar, descubra quais modelos seu LiteLLM Proxy oferece:

```bash
curl -H "Authorization: Bearer SUA_API_KEY" \
     https://flow.ciandt.com/flow-llm-proxy/v1/models
```

Use o `id` retornado como valor do `model` no código.

---

## Passo 6.3 — Atualizar `examples/agent.py`

```python
"""Agente com LLM real via LiteLLM Proxy."""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from ai_platform import track_agent

load_dotenv()

SYSTEM_PROMPT = """Você é um assistente prestativo. Responda de forma clara e concisa."""


def build_model():
    """Constrói o modelo LLM apontando para o LiteLLM Proxy."""
    model = init_chat_model(
        model="openai:gpt-4.1",              # (1) modelo disponível no seu LiteLLM
        model_provider="openai",              # (2) usa protocolo OpenAI-compatible
        base_url=os.getenv("LLM_BASE_URL"),   # (3) aponta para o LiteLLM Proxy
        api_key=os.getenv("LLM_API_KEY"),     # (4) sua API key
    )
    return model


@track_agent
def invoke_agent(message: str):
    """Ponto de entrada do agente."""
    model = build_model()
    response = model.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ])
    return {"role": "assistant", "content": response.content}


if __name__ == "__main__":
    import sys
    message = sys.argv[1] if len(sys.argv) > 1 else "Qual a capital do Brasil?"
    result = invoke_agent(message)
    print(result)
```

---

## O que cada parâmetro do `init_chat_model` faz

| Parâmetro | Valor | Explicação |
|---|---|---|
| `model` | `"openai:gpt-4.1"` | Prefixo `openai:` = provider OpenAI. `gpt-4.1` = nome do modelo no LiteLLM. |
| `model_provider` | `"openai"` | Usa protocolo OpenAI (`/v1/chat/completions`). LiteLLM é compatível. |
| `base_url` | do `.env` | URL do LiteLLM Proxy. LangChain faz `POST {base_url}/chat/completions`. |
| `api_key` | do `.env` | Enviado como `Authorization: Bearer {api_key}` no header HTTP. |

---

## O que muda no MLflow com LLM real vs Mock

| Aspecto | Com Mock | Com LLM Real |
|---|---|---|
| Input artifact | ✅ Registrado | ✅ Registrado |
| Output artifact | ✅ Registrado | ✅ Registrado |
| Tags de governança | ✅ Registradas | ✅ Registradas |
| **Traces LLM** | ❌ Não há | ✅ **Capturados pelo autolog** |
| Tokens usados | ❌ Não há | ✅ prompt_tokens, completion_tokens |
| Latência por chamada | ❌ Não há | ✅ Tempo de cada chamada LLM |
| Prompt completo | ❌ Não há | ✅ System + User messages |

Os **traces** são o grande diferencial — é o que torna o MLflow realmente útil para debugging e comparação.

---

## Troubleshooting

| Erro | Causa | Solução |
|---|---|---|
| `Connection refused` | URL errada ou LiteLLM fora do ar | Teste com `curl` primeiro |
| `401 Unauthorized` | API key inválida | Verifique a key no `.env` |
| `404 Not Found` | Endpoint errado (falta ou sobra `/v1`) | Teste `curl {URL}/v1/models` |
| `Model not found` | Nome do modelo errado | Liste modelos com `/v1/models` |

---

## Próxima etapa

➡️ [Etapa 7 — Prompt Registry: Versionando Prompts no MLflow](ETAPA_07.md)
