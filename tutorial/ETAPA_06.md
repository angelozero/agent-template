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
DOMAIN=domain_bar
ENVIROMENT=dev

# LLM Gateway (LiteLLM Proxy)
LLM_BASE_URL=sua-url-do-litellm-proxy
LLM_API_KEY=sua-api-key-aqui

MLFLOW_TRACKING_URI=http://localhost:5050
```

> **Atenção ao `/v1`**: o LangChain com `model_provider="openai"` espera que a URL termine em `/v1`, pois ele concatena `/chat/completions` no final.

---

## Passo 6.2 — Descobrir modelos disponíveis

Antes de rodar, descubra quais modelos seu LiteLLM Proxy oferece:

```bash
curl -H "Authorization: Bearer SUA_API_KEY" \
     SUA_LLM_BASE_URL/models
```

Use o `id` retornado como valor do `model` no código.

---

## Passo 6.3 — Criar `agents/agent.py`

```python
import os
import sys
import mlflow
from dotenv import load_dotenv
from config import track_agent

load_dotenv()

"""
Agente com LLM real via LiteLLM Proxy.
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from config import track_agent

load_dotenv()


def build_model():
    """Constrói o modelo LLM apontando para o LiteLLM Proxy."""
    model = init_chat_model(
        model="gpt-5",
        model_provider="openai",
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )
    return model


@track_agent
def invoke_agent(message: str):
    """Ponto de entrada do agente."""
    model = build_model()

    ### Carrega o prompt do MLflow Registry
    # "prompts:/prompt_cidades_capitais_br@latest" = última versão
    # "prompts:/prompt_cidades_capitais_br/1"      = versão específica
    # "prompts:/prompt_cidades_capitais_br@prod"   = alias (ex: produção)
    prompt = mlflow.genai.load_prompt("prompts:/prompt_cidades_capitais_br@latest")

    # Preenche as variáveis do template
    domain = os.getenv("DOMAIN", "domain_bar")
    messages = prompt.format(domain=domain, user_message=message)

    # Chamada simples: envia mensagem e recebe resposta
    response = model.invoke(messages)
    return {"role": "assistant", "content": response.content}


if __name__ == "__main__":
    message = (
        sys.argv[1]
        if len(sys.argv) > 1
       # else "Quantas capitais tem o Brasil?"
       # else "Quantas capitais tem o Brasil em relação ao Japão?"
       # else "Nas minhas viagens eu estive no oriente, em relação ao Brasil quantos continentes no oriente nós temos?"
       # else "Quantas ilhas nós temos em volta do Brasil?"
       else "A bandeira do Brasil tem estrelas que se não me enganam representam as capitais do Brasil, é igual as estrelas da bandeira dos Estados Unidos?"
    )
    result = invoke_agent(message)
    print(f"\n{result}\n")
```

> **Nota:** Este agente já integra o Prompt Registry (Etapa 7). Na primeira vez, você pode usar um prompt hardcoded e depois migrar para o Registry. Aqui mostramos a versão final.

---

## O que cada parâmetro do `init_chat_model` faz

| Parâmetro | Valor | Explicação |
|---|---|---|
| `model` | `"gpt-5"` | Nome do modelo disponível no seu LiteLLM Proxy. |
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
