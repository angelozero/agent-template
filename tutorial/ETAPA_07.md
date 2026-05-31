# Etapa 7 — Prompt Registry: Versionando Prompts no MLflow

## O que vamos aprender

- Registrar prompts como artefatos versionados no MLflow (Prompt Registry)
- Carregar prompts dinamicamente no agente (em vez de hardcoded)
- Vincular a versão do prompt usada a cada run

---

## Conceito

Até agora, o prompt está **hardcoded** no código:

```python
SYSTEM_PROMPT = """Você é um assistente prestativo..."""  # ← fixo no código
```

O problema: para mudar o prompt, você precisa **alterar código e redeployar**. Com o Prompt Registry do MLflow, o prompt vira um **artefato versionado**.

```
MLflow Prompt Registry
├── assistente-einstein
│   ├── v1 → "Você é um assistente..."
│   ├── v2 → "Você é um médico especialista..."
│   └── v3 → "Responda em formato JSON..."
│
└── Seu agent.py carrega: load_prompt("assistente-einstein@latest")
```

---

## Importante: Prompts ≠ Experiments

Prompts e Experiments são **entidades separadas** no MLflow:

```
MLflow UI (http://localhost:5000)
├── 📁 Experiments        ← seus runs ficam aqui
├── 📦 Models
├── 📝 Prompts            ← ⭐ SEUS PROMPTS FICAM AQUI (menu lateral)
└── ⚙️ Settings
```

A **associação** entre prompt e experimento acontece **automaticamente** quando você chama `mlflow.genai.load_prompt()` dentro de um `mlflow.start_run()`.

---

## Passo 7.1 — Criar o script de registro de prompts

Crie o arquivo `prompts/register_prompts.py`:

```python
"""
Script para registrar/versionar prompts no MLflow Prompt Registry.

Uso:
    python prompts/register_prompts.py

Cada execução cria uma NOVA VERSÃO do prompt (como um git commit).
"""
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()

# ⚠️ CRÍTICO: configurar o tracking URI ANTES de qualquer operação
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# ── Prompt v1: Assistente genérico ──────────────────────────────────────────
prompt_v1 = mlflow.genai.register_prompt(
    name="assistente-einstein",
    template=[
        {
            "role": "system",
            "content": (
                "Você é um assistente prestativo do Hospital Einstein. "
                "Responda de forma clara, concisa e profissional. "
                "Domínio: {{domain}}. "
                "Se não souber a resposta, diga que não sabe."
            ),
        },
        {
            "role": "user",
            "content": "{{user_message}}",
        },
    ],
    commit_message="v1: Prompt inicial do assistente genérico",
    tags={
        "author": "tutorial",
        "domain": "geral",
    },
)

print(f"✅ Prompt registrado: {prompt_v1.name} (versão {prompt_v1.version})")
```

### O que cada campo faz

| Elemento | Explicação |
|---|---|
| `name="assistente-einstein"` | Identificador único do prompt (como nome de um repositório) |
| `template=[{role, content}]` | O prompt em formato chat. Variáveis usam `{{duplas_chaves}}` |
| `{{domain}}`, `{{user_message}}` | Variáveis preenchidas em runtime com `.format()` |
| `commit_message` | Mensagem descritiva da versão (como git commit) |
| `tags` | Metadados para busca e filtragem |

---

## Passo 7.2 — Registrar uma segunda versão

Crie `prompts/register_v2.py`:

```python
"""Registra v2 do prompt com regras de formato."""
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

prompt_v2 = mlflow.genai.register_prompt(
    name="assistente-einstein",  # ← MESMO nome = nova versão
    template=[
        {
            "role": "system",
            "content": (
                "Você é um assistente especializado do Hospital Einstein. "
                "Domínio: {{domain}}. "
                "REGRAS: "
                "1. Responda SEMPRE em português brasileiro. "
                "2. Seja conciso (máximo 3 parágrafos). "
                "3. Se a pergunta for sobre saúde, inclua o disclaimer: "
                "'Esta informação é educativa e não substitui consulta médica.' "
                "4. Se não souber, diga 'Não tenho essa informação.'"
            ),
        },
        {
            "role": "user",
            "content": "{{user_message}}",
        },
    ],
    commit_message="v2: Adicionadas regras de formato e disclaimer de saúde",
    tags={
        "author": "tutorial",
        "domain": "geral",
        "change": "added-format-rules",
    },
)

print(f"✅ Prompt registrado: {prompt_v2.name} (versão {prompt_v2.version})")
```

---

## Passo 7.3 — Verificar que os prompts foram salvos

Crie `prompts/verify_prompts.py`:

```python
"""Verifica que os prompts foram registrados no MLflow."""
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Carregar a última versão
prompt = mlflow.genai.load_prompt("prompts:/assistente-einstein@latest")
print(f"Nome: {prompt.name}")
print(f"Versão: {prompt.version}")
print(f"Template: {prompt.template}")

# Carregar versão específica
prompt_v1 = mlflow.genai.load_prompt("prompts:/assistente-einstein/1")
print(f"\nVersão 1: {prompt_v1.template}")
```

---

## Passo 7.4 — Atualizar o agente para carregar prompt do Registry

Atualize `examples/agent.py`:

```python
"""Agente com prompt versionado via MLflow Prompt Registry."""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import mlflow

from ai_platform import track_agent

load_dotenv()


def build_model():
    return init_chat_model(
        model="openai:gpt-4.1",
        model_provider="openai",
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )


@track_agent
def invoke_agent(message: str):
    """Ponto de entrada — agora com prompt dinâmico."""
    model = build_model()

    # ── Carrega o prompt do MLflow Registry ──────────────────────────
    # "prompts:/assistente-einstein@latest" = última versão
    # "prompts:/assistente-einstein/1"      = versão específica
    prompt = mlflow.genai.load_prompt("prompts:/assistente-einstein@latest")

    # Preenche as variáveis do template
    domain = os.getenv("DOMAIN", "geral")
    messages = prompt.format(domain=domain, user_message=message)

    # Chama o LLM
    response = model.invoke(messages)

    # Associação manual: loga prompt usado como parâmetro e tag
    mlflow.log_param("prompt_name", prompt.name)
    mlflow.log_param("prompt_version", prompt.version)
    mlflow.set_tag("ai_platform.prompt", f"{prompt.name}/v{prompt.version}")
    mlflow.log_text(str(prompt.template), "prompts/template_used.txt")

    return {"role": "assistant", "content": response.content}


if __name__ == "__main__":
    import sys
    message = sys.argv[1] if len(sys.argv) > 1 else "O que é MLflow?"
    result = invoke_agent(message)
    print(result)
```

### Por que a associação manual?

Quando você usa `load_prompt` dentro de um `start_run`, o MLflow **automaticamente** vincula a versão do prompt ao Run. Mas para garantir visibilidade, também logamos como parâmetro e tag — assim fica visível na lista de runs sem precisar abrir cada um.

---

## Sequência de execução

```bash
# 1. MLflow rodando
just up

# 2. Registrar prompts
python prompts/register_prompts.py
# ✅ Prompt registrado: assistente-einstein (versão 1)

python prompts/register_v2.py
# ✅ Prompt registrado: assistente-einstein (versão 2)

# 3. Verificar
python prompts/verify_prompts.py

# 4. Executar o agente (carrega prompt do registry)
python examples/agent.py "O que é o Hospital Einstein?"

# 5. No MLflow UI:
#    - Menu "Prompts" → vê o prompt e suas versões
#    - Menu "Experiments" → no Run, vê params/tags com versão do prompt
```

---

## Checklist de troubleshooting

| Verificação | Comando |
|---|---|
| MLflow está rodando? | `curl http://localhost:5000/health` |
| Tracking URI configurado? | Verifique `MLFLOW_TRACKING_URI` no `.env` |
| Prompt foi registrado? | `python prompts/verify_prompts.py` |
| Prompts na UI? | Menu lateral "Prompts" no MLflow |

---

## Próxima etapa

➡️ [Etapa 8 — Evaluation: Dataset + Judges para Validar o Agente](ETAPA_08.md)
