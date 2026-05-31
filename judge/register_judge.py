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
