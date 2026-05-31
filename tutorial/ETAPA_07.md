# Etapa 7 — Prompt Registry: Versionando Prompts no MLflow

## O que vamos aprender

- Registrar prompts como artefatos versionados no MLflow (Prompt Registry)
- Carregar prompts dinamicamente no agente (em vez de hardcoded)
- Evoluir prompts incrementalmente (v1 → v4)

---

## Conceito

Até agora, o prompt poderia estar **hardcoded** no código:

```python
SYSTEM_PROMPT = """Você é um assistente..."""  # ← fixo no código
```

O problema: para mudar o prompt, você precisa **alterar código e redeployar**. Com o Prompt Registry do MLflow, o prompt vira um **artefato versionado**.

```
MLflow Prompt Registry
├── prompt_cidades_capitais_br
│   ├── v1 → "Você é um assistente com conhecimento sobre o Brasil..."
│   ├── v2 → "Você é um assistente especializado em capitais..." (+ regras)
│   ├── v3 → "..." (+ regra: nunca responder sobre outros países)
│   └── v4 → "..." (+ regra: apenas cidades/capitais do território BR)
│
└── Seu agent.py carrega: load_prompt("prompt_cidades_capitais_br@latest")
```

---

## Importante: Prompts ≠ Experiments

Prompts e Experiments são **entidades separadas** no MLflow:

```
MLflow UI (http://localhost:5050)
├── 📁 Experiments        ← seus runs ficam aqui
├── 📦 Models
├── 📝 Prompts            ← ⭐ SEUS PROMPTS FICAM AQUI (menu lateral)
└── ⚙️ Settings
```

A **associação** entre prompt e experimento acontece **automaticamente** quando você chama `mlflow.genai.load_prompt()` dentro de um `mlflow.start_run()`.

---

## Passo 7.1 — Registrar o prompt v1 (genérico)

Crie o arquivo `prompt/register_prompt_v1.py`:

```python
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

prompt_v1 = mlflow.genai.register_prompt(
    name="prompt_cidades_capitais_br",
    template=[
        {
            "role": "system",
            "content": (
                "Você é um assistente com conhecimento sobre o Brasil"
                "Responda de forma clara, concisa e profissional"
                "Domínio: {{domain}}."
                "Se não souber a resposta, diga que não sabe"
            ),
        },
        {"role": "user", "content": "{{user_message}}"},
    ],
    commit_message="V1: Prompt Capitas Brasileiras",
    tags={"author": "tutorial", "domain": "geral"},
)

print(f"\n✅ Prompt registrado com sucesso - prompt: {prompt_v1.name} - version: {prompt_v1.version}\n")
```

### O que cada campo faz

| Elemento | Explicação |
|---|---|
| `name="prompt_cidades_capitais_br"` | Identificador único do prompt (como nome de um repositório) |
| `template=[{role, content}]` | O prompt em formato chat. Variáveis usam `{{duplas_chaves}}` |
| `{{domain}}`, `{{user_message}}` | Variáveis preenchidas em runtime com `.format()` |
| `commit_message` | Mensagem descritiva da versão (como git commit) |
| `tags` | Metadados para busca e filtragem |

---

## Passo 7.2 — Registrar o prompt v2 (com regras de formato)

Crie `prompt/register_prompt_v2.py`:

```python
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

prompt_v2 = mlflow.genai.register_prompt(
    # se 'name' estiver com o mesmo valor de prompt_v1 ele será sobrescrito mas manterá os versionamentos
    name="prompt_cidades_capitais_br",
    template=[
        {
            "role": "system",
            "content": (
                "Você é um assistente especializado em capitais e cidades brasileiras. "
                "Domínio: {{domain}}. "
                "REGRAS: "
                "1. Responda SEMPRE em português brasileiro. "
                "2. Seja conciso (máximo 3 parágrafos). "
                "3. Se a pergunta for sobre qualquer outro assunto que não seja capitais ou cidades brasileiras, inclua o disclaimer: "
                "'Não tenho conhecimento sobre o assunto.' "
                "4. Se não souber, diga 'Não tenho essa informação.'"
            ),
        },
        {
            "role": "user",
            "content": "{{user_message}}",
        },
    ],
    commit_message="V2: Prompt Capitas Brasileiras",
    tags={
        "author": "tutorial",
        "domain": "geral",
        "change": "added-format-rules",
    },
)

print(f"\n✅ Prompt registrado com sucesso - prompt: {prompt_v2.name} - version: {prompt_v2.version}\n")
```

---

## Passo 7.3 — Registrar o prompt v3 (restrição a Brasil)

Crie `prompt/register_prompt_v3.py`:

```python
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

prompt_v2 = mlflow.genai.register_prompt(
    name="prompt_cidades_capitais_br",
    template=[
        {
            "role": "system",
            "content": (
                "Você é um assistente especializado em capitais e cidades brasileiras. "
                "Domínio: {{domain}}. "
                "REGRAS: "
                "1. Responda SEMPRE em português brasileiro. "
                "2. Seja conciso (máximo 3 parágrafos). "
                "3. Se a pergunta for sobre qualquer outro assunto que não seja capitais ou cidades brasileiras, inclua o disclaimer: "
                "'Não tenho conhecimento sobre o assunto.' "
                "4. Se não souber, diga 'Não tenho essa informação.'"
                "5. Nunca responda nada que não seja unica e exclusivamente sobre o Brasil"
            ),
        },
        {
            "role": "user",
            "content": "{{user_message}}",
        },
    ],
    commit_message="V2: Prompt Capitas Brasileiras",
    tags={
        "author": "tutorial",
        "domain": "geral",
        "change": "added-format-rules",
    },
)

print(f"\n✅ Prompt registrado com sucesso - prompt: {prompt_v2.name} - version: {prompt_v2.version}\n")
```

---

## Passo 7.4 — Registrar o prompt v4 (restrição a cidades/capitais do território BR)

Crie `prompt/register_prompt_v4.py`:

```python
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

prompt_v2 = mlflow.genai.register_prompt(
    name="prompt_cidades_capitais_br",
    template=[
        {
            "role": "system",
            "content": (
                "Você é um assistente especializado em capitais e cidades brasileiras. "
                "Domínio: {{domain}}. "
                "REGRAS: "
                "1. Responda SEMPRE em português brasileiro. "
                "2. Seja conciso (máximo 3 parágrafos). "
                "3. Se a pergunta for sobre qualquer outro assunto que não seja capitais ou cidades brasileiras, inclua o disclaimer: "
                "'Não tenho conhecimento sobre o assunto.' "
                "4. Se não souber, diga 'Não tenho essa informação.'"
                "5. Nunca responda nada que não seja unica e exclusivamente sobre cidades ou capitais dentro do território brasileiro"
            ),
        },
        {
            "role": "user",
            "content": "{{user_message}}",
        },
    ],
    commit_message="V2: Prompt Capitas Brasileiras",
    tags={
        "author": "tutorial",
        "domain": "geral",
        "change": "added-format-rules",
    },
)

print(f"\n✅ Prompt registrado com sucesso - prompt: {prompt_v2.name} - version: {prompt_v2.version}\n")
```

---

## Evolução dos prompts: v1 → v4

| Versão | O que mudou | Por quê |
|---|---|---|
| **v1** | Prompt genérico sobre o Brasil | Ponto de partida — muito amplo |
| **v2** | Adicionou regras de formato (idioma, concisão, disclaimer, fallback) | Agente respondia em inglês, sem limites |
| **v3** | Adicionou regra 5: "nunca responder sobre outros países" | Agente ainda comparava Brasil com outros países |
| **v4** | Refinou regra 5: "apenas cidades/capitais do território brasileiro" | Agente respondia sobre geografia física, economia, etc. |

Cada versão é uma **iteração** baseada em testes reais. Na Etapa 8, veremos como usar Judges para avaliar automaticamente se o prompt está funcionando.

---

## Passo 7.5 — Atualizar o agente para carregar prompt do Registry

O `agents/agent.py` já carrega o prompt dinamicamente:

```python
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
```

### Formas de carregar um prompt

| URI | O que carrega |
|---|---|
| `prompts:/prompt_cidades_capitais_br@latest` | Última versão registrada |
| `prompts:/prompt_cidades_capitais_br/1` | Versão específica (v1) |
| `prompts:/prompt_cidades_capitais_br@prod` | Alias (ex: produção) |

---

## Sequência de execução

```bash
# 1. MLflow rodando
docker compose -f docker/docker-compose.yml up -d

# 2. Registrar prompts (execute cada um na ordem)
uv run python prompt/register_prompt_v1.py
# ✅ Prompt registrado com sucesso - prompt: prompt_cidades_capitais_br - version: 1

uv run python prompt/register_prompt_v2.py
# ✅ Prompt registrado com sucesso - prompt: prompt_cidades_capitais_br - version: 2

uv run python prompt/register_prompt_v3.py
# ✅ Prompt registrado com sucesso - prompt: prompt_cidades_capitais_br - version: 3

uv run python prompt/register_prompt_v4.py
# ✅ Prompt registrado com sucesso - prompt: prompt_cidades_capitais_br - version: 4

# 3. Executar o agente (carrega prompt do registry)
uv run python agents/agent.py "Qual a capital do Brasil?"

# 4. No MLflow UI (http://localhost:5050):
#    - Menu "Prompts" → vê o prompt e suas 4 versões
#    - Menu "Experiments" → no Run, vê o prompt usado
```

---

## Checklist de troubleshooting

| Verificação | Comando |
|---|---|
| MLflow está rodando? | `curl http://localhost:5050/health` |
| Tracking URI configurado? | Verifique `MLFLOW_TRACKING_URI` no `.env` |
| Prompt foi registrado? | Verifique no menu "Prompts" do MLflow UI |

---

## Próxima etapa

➡️ [Etapa 8 — Evaluation: Dataset + Judges para Validar o Agente](ETAPA_08.md)
