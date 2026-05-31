# Etapa 8 — Evaluation: Dataset + Judges para Validar o Agente

## O que vamos aprender

- Criar um dataset de avaliação (inputs + outputs esperados)
- Configurar LLM Judges para usar o LiteLLM Proxy
- Usar **LLM Judges** (scorers) para avaliar automaticamente as respostas
- Comparar qualidade entre versões de prompts/modelos

---

## Conceito

Avaliação responde à pergunta: **"Meu agente está respondendo bem?"**

```
Dataset de Avaliação          Seu Agente          Judges (Avaliadores)
┌──────────────────┐     ┌──────────────┐     ┌─────────────────────────┐
│ Q: Capital do BR?│────▶│ invoke_agent │────▶│ Correctness             │
│ Esperado: ...    │     │              │     │ "Resposta correta?"     │
├──────────────────┤     │              │     ├─────────────────────────┤
│ Q: Capital Japão?│────▶│              │────▶│ RelevanceToQuery        │
│ Esperado: ...    │     │              │     │ "Resposta relevante?"   │
├──────────────────┤     │              │     ├─────────────────────────┤
│ Q: Receita bolo? │────▶│              │────▶│ Guidelines              │
│ Esperado: ...    │     │              │     │ "Segue as regras?"      │
└──────────────────┘     └──────────────┘     └─────────────────────────┘
                                                        │
                                                        ▼
                                              📊 Relatório no MLflow
```

---

## Passo 8.1 — Criar o dataset de avaliação

Crie `dataset/register_dataset.py`:

```python
"""
Dataset de avaliação para o agente de capitais e cidades brasileiras.

Baseado nas regras definidas nos prompts (v1–v4):
  1. Responder SEMPRE em português brasileiro
  2. Ser conciso (máximo 3 parágrafos)
  3. Fora do escopo (não é cidade/capital BR) → disclaimer: "Não tenho conhecimento sobre o assunto."
  4. Não souber → "Não tenho essa informação."
  5. Nunca responder nada que não seja sobre cidades/capitais dentro do território brasileiro

Cada item tem:
  inputs:       a pergunta do usuário
  expectations: o que esperamos da resposta (fatos, resposta esperada)
"""

EVAL_DATASET = [
    # ═══════════════════════════════════════════════════════════════════════
    # GRUPO 1: Perguntas factuais sobre capitais brasileiras (DENTRO do escopo)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Caso 1: Capital federal ───────────────────────────────────────────
    {
        "inputs": {"query": "Qual a capital do Brasil?"},
        "expectations": {
            "expected_facts": [
                "A capital do Brasil é Brasília",
            ],
            "expected_response": "Brasília é a capital do Brasil, localizada no Distrito Federal.",
        },
    },
    # ── Caso 2: Capital de estado (região Sudeste) ────────────────────────
    {
        "inputs": {"query": "Qual a capital de São Paulo?"},
        "expectations": {
            "expected_facts": [
                "A capital do estado de São Paulo é a cidade de São Paulo",
            ],
            "expected_response": "A capital do estado de São Paulo é a cidade de São Paulo.",
        },
    },

    # ... (19 casos de teste no total, organizados em 5 grupos)

    # ═══════════════════════════════════════════════════════════════════════
    # GRUPO 3: Perguntas FORA do escopo (deve retornar disclaimer)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Caso 8: País estrangeiro ──────────────────────────────────────────
    {
        "inputs": {"query": "Qual a cidade mais populosa do Japão?"},
        "expectations": {
            "expected_facts": [
                "A resposta deve conter o disclaimer de que não tem conhecimento sobre o assunto",
                "O agente NÃO deve responder sobre o Japão",
            ],
            "expected_response": "Não tenho conhecimento sobre o assunto.",
        },
    },

    # ... mais casos de teste
]
```

### Estrutura do dataset

O dataset está organizado em **5 grupos**:

| Grupo | Casos | O que testa |
|---|---|---|
| **1. Capitais brasileiras** | 1–5 | Perguntas factuais sobre capitais (dentro do escopo) |
| **2. Cidades brasileiras** | 6–7 | Perguntas sobre cidades BR (dentro do escopo) |
| **3. Fora do escopo** | 8–13 | Perguntas que devem retornar disclaimer |
| **4. Idioma** | 14–15 | Perguntas em inglês/espanhol (deve responder em PT-BR) |
| **5. Casos-limite** | 16–19 | Perguntas mistas, ambíguas, tentativas de desvio |

Cada item tem:
- `inputs.query`: a pergunta do usuário
- `expectations.expected_facts`: fatos que a resposta deve conter
- `expectations.expected_response`: resposta esperada (referência)

---

## Passo 8.2 — Configurar os Judges para usar o LiteLLM Proxy

Crie `config/judge_config.py`:

```python
"""
Configuração dos LLM Judges para avaliação via MLflow.

Responsabilidades:
1. Mapeia LLM_API_KEY → OPENAI_API_KEY (autenticação dos judges)
2. Redireciona os judges para o LiteLLM Proxy (em vez de api.openai.com)
3. Define o modelo padrão dos judges

Uso:
    from config.judge_config import JUDGE_MODEL, setup_judge_provider
    setup_judge_provider()
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── Modelo usado pelos LLM Judges ────────────────────────────────────────────
# Deve estar disponível no LiteLLM Proxy. Formato: "openai:/<model_name>"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai:/gpt-5")


def setup_judge_provider():
    """
    Configura o ambiente para que os LLM Judges do MLflow usem o LiteLLM Proxy.

    Os scorers (Correctness, RelevanceToQuery, Guidelines) usam a API OpenAI
    internamente. Esta função:
    - Mapeia LLM_API_KEY → OPENAI_API_KEY
    - Aplica um patch no MLflow para redirecionar as chamadas do provider OpenAI
      para o LLM_BASE_URL (LiteLLM Proxy), já que o MLflow hardcodes
      "https://api.openai.com/v1" como endpoint padrão.
    """
    _setup_openai_api_key()
    _patch_openai_provider()


def _setup_openai_api_key():
    """Mapeia LLM_API_KEY → OPENAI_API_KEY se não estiver definida."""
    if os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("LLM_API_KEY")


def _patch_openai_provider():
    """
    Patch no MLflow para redirecionar o provider OpenAI para o LiteLLM Proxy.

    O MLflow cria o OpenAIConfig sem openai_api_base, fazendo os judges chamarem
    api.openai.com diretamente. Este patch intercepta a criação do provider e
    injeta o LLM_BASE_URL como openai_api_base.
    """
    llm_base_url = os.getenv("LLM_BASE_URL")
    if not llm_base_url:
        return

    import mlflow.metrics.genai.model_utils as _model_utils
    from mlflow.gateway.config import EndpointConfig, OpenAIConfig, Provider
    from mlflow.gateway.providers.openai import OpenAIProvider

    _original_get_provider = _model_utils._get_provider_instance

    def _patched_get_provider(provider, model, base_url=None):
        """Injeta LLM_BASE_URL como openai_api_base para o provider OpenAI."""
        if provider == Provider.OPENAI:
            config = OpenAIConfig(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                openai_api_base=llm_base_url.rstrip("/"),
            )
            route_config = EndpointConfig(
                name=provider,
                endpoint_type="llm/v1/chat",
                model={
                    "provider": provider,
                    "name": model,
                    "config": config.model_dump(),
                },
            )
            return OpenAIProvider(route_config)
        return _original_get_provider(provider, model, base_url=base_url)

    _model_utils._get_provider_instance = _patched_get_provider
```

### Por que o patch é necessário?

Os LLM Judges do MLflow (`Correctness`, `RelevanceToQuery`, `Guidelines`) usam a API OpenAI internamente para avaliar as respostas. Por padrão, eles chamam `https://api.openai.com/v1`. Se você usa um **LiteLLM Proxy** como gateway, precisa redirecionar essas chamadas.

O `setup_judge_provider()` faz duas coisas:
1. **Mapeia a API key**: `LLM_API_KEY` → `OPENAI_API_KEY`
2. **Redireciona o endpoint**: intercepta a criação do provider OpenAI e injeta o `LLM_BASE_URL`

---

## Passo 8.3 — Criar o script de avaliação com Judges

Crie `judge/register_judge.py`:

```python
"""
Avaliação do agente usando MLflow GenAI Evaluate + LLM Judges.

Uso:
    python judge/register_judge.py

O que faz:
1. Carrega o dataset de avaliação
2. Executa o agente para cada pergunta
3. Usa LLM Judges para avaliar cada resposta
4. Registra os resultados no MLflow
"""
import os
import mlflow
from dotenv import load_dotenv
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines

from config.judge_config import JUDGE_MODEL, setup_judge_provider
from dataset.register_dataset import EVAL_DATASET

load_dotenv()

# ── Configura o provider dos judges (LiteLLM Proxy) ──────────────────────────
setup_judge_provider()

# ── Configura o MLflow ────────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

team = os.getenv("TEAM_NAME", "team_foo")
domain = os.getenv("DOMAIN", "domain_bar")
agent_name = os.getenv("AGENT_NAME", "agent_zero")
mlflow.set_experiment(f"/{team}/{domain}/{agent_name}/evaluations")


# ── Define a função que será avaliada
# O mlflow.genai.evaluate vai chamar esta função para cada item do dataset
def predict_fn(query: str) -> dict:
    """Wrapper que chama o agente e retorna no formato esperado pelo evaluate."""
    from agents.agent import build_model

    model = build_model()
    prompt = mlflow.genai.load_prompt("prompts:/prompt_cidades_capitais_br@latest")

    domain = os.getenv("DOMAIN", "domain_bar")
    messages = prompt.format(domain=domain, user_message=query)

    response = model.invoke(messages)
    return {"response": response.content}


# ── Define os Judges (avaliadores)
scorers = [
    # Judge 1: Correctness — verifica se a resposta contém os fatos esperados
    Correctness(model=JUDGE_MODEL),

    # Judge 2: RelevanceToQuery — verifica se a resposta é relevante à pergunta
    RelevanceToQuery(model=JUDGE_MODEL),

    # Judge 3: Guidelines — verifica regras customizadas
    Guidelines(
        name="format_rules",
        model=JUDGE_MODEL,
        guidelines=(
            "A resposta DEVE estar em português brasileiro. "
            "A resposta DEVE ter no máximo 3 parágrafos. "
            "Se a pergunta for sobre qualquer assunto sem ser cidades ou capitais brasileiras, DEVE conter o disclaimer: "
            "'Não tenho conhecimento sobre esta informação.'"
        ),
    ),
]


# ── Executa a avaliação
print("🔍 Iniciando avaliação...")
print(f"   Dataset: {len(EVAL_DATASET)} casos de teste")
print(f"   Judges: {[s.__class__.__name__ for s in scorers]}")
print()

results = mlflow.genai.evaluate(
    data=EVAL_DATASET,
    predict_fn=predict_fn,  # função que gera as respostas
    scorers=scorers,        # judges que avaliam as respostas
)

# ── Exibe resultados 
print("\n📊 Resultados da avaliação:")
print("─" * 52)
for metric_name, metric_value in results.metrics.items():
    print(f"  {metric_name}: {metric_value}")
print("─" * 52)
print(f"\n🔗 Veja detalhes no MLflow UI: http://localhost:5050")
```

---

## Como cada Judge funciona

| Judge | O que avalia | Precisa de... | Analogia |
|---|---|---|---|
| `Correctness(model=JUDGE_MODEL)` | Fatos estão corretos? | `expected_facts` ou `expected_response` no dataset | Prova com gabarito |
| `RelevanceToQuery(model=JUDGE_MODEL)` | Resposta é relevante à pergunta? | Apenas `inputs.query` | "Respondeu o que foi perguntado?" |
| `Guidelines(model=JUDGE_MODEL)` | Segue regras customizadas? | Suas regras em texto livre | Checklist de QA |

### Importante: `model=JUDGE_MODEL`

Todos os scorers recebem `model=JUDGE_MODEL` para usar o modelo configurado no `config/judge_config.py`. O formato é `"openai:/<model_name>"` (ex: `"openai:/gpt-5"`).

### Importante: `predict_fn(query: str)`

A função `predict_fn` recebe diretamente a `query` como string (não um dict). O MLflow extrai o valor de `inputs.query` do dataset e passa como argumento.

---

## Sequência de execução

```bash
# 1. MLflow rodando
docker compose -f docker/docker-compose.yml up -d

# 2. Prompts registrados (etapa anterior)
uv run python prompt/register_prompt_v4.py

# 3. Executar avaliação
uv run python judge/register_judge.py

# Saída esperada:
# 🔍 Iniciando avaliação...
#    Dataset: 19 casos de teste
#    Judges: ['Correctness', 'RelevanceToQuery', 'Guidelines']
#
# 📊 Resultados da avaliação:
# ────────────────────────────────────────────────────
#   correctness/mean: 0.8
#   relevance_to_query/mean: 1.0
#   format_rules/mean: 0.6
# ────────────────────────────────────────────────────
#
# 🔗 Veja detalhes no MLflow UI: http://localhost:5050
```

---

## O que você verá no MLflow UI

No experimento `/<time>/<domínio>/<agente>/evaluations`:

- Uma tabela com cada caso de teste
- Score de cada judge para cada caso
- Justificativa do judge (por que deu pass/fail)
- Métricas agregadas (média de cada judge)

---

## Estrutura final do projeto

```
agent-template/
├── config/
│   ├── __init__.py            # expõe track_agent
│   ├── app_config.py          # Config dataclass + load_config()
│   ├── judge_config.py        # configuração dos LLM Judges
│   └── tracking.py            # @track_agent decorator
│
├── agents/
│   ├── agent_mock.py          # agente com LLM mockado
│   └── agent.py               # agente com LLM real + prompt dinâmico
│
├── prompt/
│   ├── register_prompt_v1.py  # prompt genérico
│   ├── register_prompt_v2.py  # + regras de formato
│   ├── register_prompt_v3.py  # + restrição a Brasil
│   └── register_prompt_v4.py  # + restrição a cidades/capitais BR
│
├── dataset/
│   └── register_dataset.py    # 19 casos de teste em 5 grupos
│
├── judge/
│   └── register_judge.py      # avaliação com LLM Judges
│
├── docker/
│   └── docker-compose.yml     # MLflow server local
│
├── tutorial/
│   ├── README.md
│   ├── ETAPA_01.md → ETAPA_08.md
│
├── .env_example
├── .gitignore
└── pyproject.toml
```

---

## Mapa evolutivo completo

```
✅ Etapa 1 — Estrutura + pyproject.toml
    ↓
✅ Etapa 2 — Config (.env + dataclass)
    ↓
✅ Etapa 3 — @track_agent + MLflow (o coração)
    ↓
✅ Etapa 4 — Docker Compose (MLflow local)
    ↓
✅ Etapa 5 — Agente com Mock
    ↓
✅ Etapa 6 — LLM Real (LiteLLM Proxy)
    ↓
✅ Etapa 7 — Prompt Registry (versionamento v1→v4)
    ↓
✅ Etapa 8 — Evaluation + Judges
    ↓
⏭️ Próximos passos:
    ├── Tools + LangGraph (agentes multi-step)
    ├── Custom Scorers (avaliadores personalizados)
    └── CI/CD com avaliação automática
```
