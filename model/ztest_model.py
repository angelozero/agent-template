"""
Fluxo completo de envio, recuperação e teste do modelo MLflow.

Ordem de execução:
  1. send_model()  → empacota agents/agent.py e registra no MLflow Model Registry
  2. test_model()  → baixa o modelo registrado e executa .predict()

Uso:
    uv run python model/ztest_model.py
"""

import sys
from pathlib import Path

# Garante que a raiz do projeto está no sys.path independente de como o arquivo
# é executado (python model/ztest_model.py ou uv run python model/ztest_model.py).
# Deve ficar ANTES de qualquer import de módulos do projeto (model, wrapper, config, etc).
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mlflow
import pandas as pd
from dotenv import load_dotenv

from model.load_model import load_model
from model.register_model import register_model

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# 1. send_model — empacota e envia o modelo ao MLflow
#    Deve ser chamado ANTES de load_model() e test_model()
# ══════════════════════════════════════════════════════════════════════════════

def send_model(agent_file: str) -> mlflow.models.model.ModelInfo:
    """
    Empacota o agente e registra no MLflow Model Registry.

    Deve ser chamado ANTES de test_model(), pois o modelo
    precisa existir no Registry para poder ser recuperado.

    Delega toda a lógica de empacotamento para model/register_model.py:
      - Valida e resolve o caminho absoluto do agente
      - Busca a última run dev para rastreabilidade
      - Loga o modelo com AgentPyfuncWrapper + assinatura + artefatos
      - Registra no Model Registry com nome <team>.<domain>.<agent>

    Args:
        agent_file: Caminho para o arquivo do agente (obrigatório).
                    Exemplo: "agents/agent.py"

    Returns:
        ModelInfo com model_uri do modelo registrado.
    """
    print("\n" + "═" * 56)
    print("  📦 ETAPA 1 — ENVIANDO MODELO PARA O MLFLOW")
    print("═" * 56 + "\n")

    model_info = register_model(agent_file=agent_file)
    return model_info


# ══════════════════════════════════════════════════════════════════════════════
# 2. test_model — carrega o modelo do MLflow e executa .predict()
#    Deve ser chamado APÓS send_model()
# ══════════════════════════════════════════════════════════════════════════════

def test_model(
    model_uri: str | None = None,
    message: str = "Qual a capital do Brasil?",
) -> pd.DataFrame:
    """
    Carrega o modelo do MLflow e executa uma predição de teste.

    Fluxo:
      1. Chama load_model() para baixar e inicializar o modelo
      2. Monta o input como pd.DataFrame com coluna "message"
      3. Chama .predict() no modelo carregado
      4. Exibe e retorna o resultado

    Args:
        model_uri: URI do modelo no MLflow. Se None, usa a última versão registrada.
                   Exemplos:
                   - "models:/<team>.<domain>.<agent>/latest"
                   - "models:/<team>.<domain>.<agent>/1"
                   - "runs:/<run_id>/agent"
        message:   Mensagem de teste enviada ao agente.

    Returns:
        pd.DataFrame com coluna "answer" contendo a resposta do agente.
    """
    print("\n" + "═" * 56)
    print("  🔽 ETAPA 2 — RECUPERANDO MODELO DO MLFLOW")
    print("═" * 56 + "\n")

    # ── Carrega o modelo ──────────────────────────────────────────────────────
    loaded_model = load_model(model_uri=model_uri)

    # ── Monta o input ─────────────────────────────────────────────────────────
    input_df = pd.DataFrame([{"message": message}])

    print("═" * 56)
    print("  🧪 ETAPA 3 — EXECUTANDO AGENTE SALVO NO MLFLOW")
    print("═" * 56)
    print(f"  Mensagem   : {message}")
    print("─" * 56)

    # ── Executa a predição ────────────────────────────────────────────────────
    # .predict() chama AgentPyfuncWrapper.predict(), que por sua vez chama
    # invoke_agent() do agents/agent.py carregado dinamicamente.
    result = loaded_model.predict(input_df)

    # ── Exibe o resultado ─────────────────────────────────────────────────────
    answer = result["answer"].iloc[0]

    print(f"\n  📤 RESPOSTA DO AGENTE:")
    print(f"  {answer}")
    print("═" * 56 + "\n")

    return result


if __name__ == "__main__":
    # 1. Empacota o agente e registra no MLflow Model Registry
    model_info = send_model(agent_file="agents/agent.py")

    # 2. Baixa o modelo registrado e testa chamando .predict()
    test_model(
        model_uri=model_info.model_uri,
        message="Qual o valor total de capitais do Brasil?",
    )
