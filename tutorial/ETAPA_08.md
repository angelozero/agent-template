# Etapa 8 — Evaluation: Dataset + Judges para Validar o Agente

## O que vamos aprender

- Criar um dataset de avaliação (inputs + outputs esperados)
- Usar **LLM Judges** (scorers) para avaliar automaticamente as respostas
- Comparar qualidade entre versões de prompts/modelos

---

## Conceito

Avaliação responde à pergunta: **"Meu agente está respondendo bem?"**

```
Dataset de Avaliação          Seu Agente          Judges (Avaliadores)
┌──────────────────┐     ┌──────────────┐     ┌─────────────────────────┐
│ Q: O que é MLflow│────▶│ invoke_agent │────▶│ Correctness             │
│ Esperado: ...    │     │              │     │ "Resposta correta?"     │
├──────────────────┤     │              │     ├─────────────────────────┤
│ Q: Telefone?     │────▶│              │────▶│ RelevanceToQuery        │
│ Esperado: ...    │     │              │     │ "Resposta relevante?"   │
├──────────────────┤     │              │     ├─────────────────────────┤
│ Q: Sintomas?     │────▶│              │────▶│ Guidelines              │
│ Esperado: ...    │     │              │     │ "Segue as regras?"      │
└──────────────────┘     └──────────────┘     └─────────────────────────┘
                                                        │
                                                        ▼
                                              📊 Relatório no MLflow
```

---

## Passo 8.1 — Criar o dataset de avaliação

Crie `evaluation/__init__.py`:

```python
# vazio — marca como pacote Python
```

Crie `evaluation/dataset.py`:

```python
"""
Dataset de avaliação para o agente assistente-einstein.

Cada item tem:
- inputs: a pergunta do usuário
- expectations: o que esperamos da resposta (fatos, resposta esperada)
"""

EVAL_DATASET = [
    # ── Caso 1: Pergunta factual simples ─────────────────────────────────
    {
        "inputs": {"query": "O que é o MLflow?"},
        "expectations": {
            "expected_facts": [
                "MLflow é uma plataforma open-source",
                "MLflow é usado para gerenciar experimentos de machine learning",
            ],
        },
    },
    # ── Caso 2: Pergunta que o agente NÃO deve saber ────────────────────
    {
        "inputs": {"query": "Qual o número de telefone do Hospital Einstein?"},
        "expectations": {
            "expected_facts": [
                "O agente deve indicar que não tem essa informação",
            ],
        },
    },
    # ── Caso 3: Pergunta de saúde (deve ter disclaimer) ─────────────────
    {
        "inputs": {"query": "Quais os sintomas de gripe?"},
        "expectations": {
            "expected_facts": [
                "Febre é um sintoma de gripe",
                "A resposta deve conter disclaimer sobre consulta médica",
            ],
        },
    },
    # ── Caso 4: Pergunta em outro idioma (deve responder em PT-BR) ──────
    {
        "inputs": {"query": "What is the capital of Brazil?"},
        "expectations": {
            "expected_facts": [
                "A capital do Brasil é Brasília",
                "A resposta deve estar em português",
            ],
        },
    },
    # ── Caso 5: Pergunta que exige concisão ──────────────────────────────
    {
        "inputs": {"query": "Resuma o que é inteligência artificial em uma frase."},
        "expectations": {
            "expected_response": (
                "Inteligência artificial é a capacidade de máquinas "
                "realizarem tarefas que normalmente requerem inteligência humana."
            ),
        },
    },
]
```

---

## Passo 8.2 — Criar o script de avaliação com Judges

Crie `evaluation/evaluate.py`:

```python
"""
Avaliação do agente usando MLflow GenAI Evaluate + LLM Judges.

Uso:
    python evaluation/evaluate.py

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

from evaluation.dataset import EVAL_DATASET

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# ── Configura o experimento de avaliação ─────────────────────────────────────
team = os.getenv("TEAM_NAME", "meu-time")
domain = os.getenv("DOMAIN", "geral")
agent_name = os.getenv("AGENT_NAME", "meu-agente")
mlflow.set_experiment(f"/{team}/{domain}/{agent_name}/evaluations")


# ── Define a função que será avaliada ────────────────────────────────────────
def predict_fn(inputs: dict) -> dict:
    """Wrapper que chama o agente e retorna no formato esperado."""
    from langchain.chat_models import init_chat_model

    model = init_chat_model(
        model="openai:gpt-4.1",
        model_provider="openai",
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )

    prompt = mlflow.genai.load_prompt("prompts:/assistente-einstein@latest")
    domain = os.getenv("DOMAIN", "geral")
    messages = prompt.format(domain=domain, user_message=inputs["query"])

    response = model.invoke(messages)
    return {"response": response.content}


# ── Define os Judges (avaliadores) ───────────────────────────────────────────
scorers = [
    # Judge 1: Correctness — verifica se a resposta contém os fatos esperados
    Correctness(),

    # Judge 2: RelevanceToQuery — verifica se a resposta é relevante à pergunta
    RelevanceToQuery(),

    # Judge 3: Guidelines — verifica regras customizadas
    Guidelines(
        name="format_rules",
        guidelines=(
            "A resposta DEVE estar em português brasileiro. "
            "A resposta DEVE ter no máximo 3 parágrafos. "
            "Se a pergunta for sobre saúde, DEVE conter o disclaimer: "
            "'Esta informação é educativa e não substitui consulta médica.'"
        ),
    ),
]


# ── Executa a avaliação ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Iniciando avaliação...")
    print(f"   Dataset: {len(EVAL_DATASET)} casos de teste")
    print(f"   Judges: {[s.__class__.__name__ for s in scorers]}")
    print()

    results = mlflow.genai.evaluate(
        data=EVAL_DATASET,
        predict_fn=predict_fn,
        scorers=scorers,
    )

    print("\n📊 Resultados da avaliação:")
    print("─" * 52)
    for metric_name, metric_value in results.metrics.items():
        print(f"  {metric_name}: {metric_value}")
    print("─" * 52)
    print(f"\n🔗 Detalhes no MLflow UI: http://localhost:5000")
```

---

## Como cada Judge funciona

| Judge | O que avalia | Precisa de... | Analogia |
|---|---|---|---|
| `Correctness()` | Fatos estão corretos? | `expected_facts` ou `expected_response` no dataset | Prova com gabarito |
| `RelevanceToQuery()` | Resposta é relevante à pergunta? | Apenas `inputs.query` | "Respondeu o que foi perguntado?" |
| `Guidelines()` | Segue regras customizadas? | Suas regras em texto livre | Checklist de QA |

### Importante: os Judges usam um LLM para avaliar

Os judges chamam um LLM para fazer a avaliação. Para que funcionem com seu LiteLLM, configure no `.env`:

```env
OPENAI_API_KEY=sua-api-key
OPENAI_API_BASE=https://flow.ciandt.com/flow-llm-proxy/v1
```

Ou especifique o modelo explicitamente:

```python
Correctness(model="openai:/gpt-4.1")
```

---

## Sequência de execução

```bash
# 1. MLflow rodando
just up

# 2. Prompts registrados (etapa anterior)
python prompts/register_prompts.py

# 3. Executar avaliação
python evaluation/evaluate.py

# Saída esperada:
# 🔍 Iniciando avaliação...
#    Dataset: 5 casos de teste
#    Judges: ['Correctness', 'RelevanceToQuery', 'Guidelines']
#
# 📊 Resultados da avaliação:
# ────────────────────────────────────────────────────
#   correctness/mean: 0.8
#   relevance_to_query/mean: 1.0
#   format_rules/mean: 0.6
# ────────────────────────────────────────────────────
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
meu-agente/
├── ai_platform/
│   ├── __init__.py
│   ├── config.py
│   └── tracking.py
│
├── examples/
│   └── agent.py              # agente com prompt dinâmico
│
├── prompts/
│   ├── register_prompts.py   # registra v1
│   ├── register_v2.py        # registra v2
│   └── verify_prompts.py     # verifica prompts
│
├── evaluation/
│   ├── __init__.py
│   ├── dataset.py             # casos de teste
│   └── evaluate.py            # executa avaliação com judges
│
├── tutorial/
│   ├── ETAPA_01.md
│   ├── ETAPA_02.md
│   ├── ETAPA_03.md
│   ├── ETAPA_04.md
│   ├── ETAPA_05.md
│   ├── ETAPA_06.md
│   ├── ETAPA_07.md
│   └── ETAPA_08.md
│
├── docker-compose.yaml
├── .env
├── .env.example
├── pyproject.toml
└── Justfile
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
✅ Etapa 4 — Docker MLflow + Justfile
    ↓
✅ Etapa 5 — Agente com Mock
    ↓
✅ Etapa 6 — LLM Real (LiteLLM Proxy)
    ↓
✅ Etapa 7 — Prompt Registry (versionamento)
    ↓
✅ Etapa 8 — Evaluation + Judges
    ↓
⏭️ Próximos passos:
    ├── Tools + LangGraph (agentes multi-step)
    ├── Custom Scorers (avaliadores personalizados)
    └── CI/CD com avaliação automática
```
