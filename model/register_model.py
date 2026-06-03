"""
Empacota agents/agent.py como modelo MLflow PyFunc e registra no Model Registry.

Uso:
    uv run python model/register_model.py
    uv run python model/register_model.py --agent-file agents/agent.py
"""

import argparse
import os
import sys
from pathlib import Path

# Garante que a raiz do projeto está no sys.path independente de como o arquivo
# é executado (python model/register_model.py ou uv run python model/register_model.py).
# Deve ficar ANTES de qualquer import de módulos do projeto (wrapper, config, etc).
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature

from model.wrapper_model import AgentPyfuncWrapper

load_dotenv()


# ── Helpers de configuração ───────────────────────────────────────────────────

def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")


def _agent_name() -> str:
    name = os.getenv("AGENT_NAME")
    if not name:
        raise EnvironmentError("AGENT_NAME não definido no .env")
    return name


def _team_name() -> str:
    return os.getenv("TEAM_NAME", "default")


def _domain() -> str:
    return os.getenv("DOMAIN", "geral")


def _artifact_path() -> str:
    """
    Nome do diretório do artefato dentro da run MLflow.
    Lido da variável ARTIFACT_PATH no .env.
    Exemplo: "agente-do-angelo"
    """
    path = os.getenv("ARTIFACT_PATH")
    if not path:
        raise EnvironmentError("ARTIFACT_PATH não definido no .env")
    return path


def registered_model_name() -> str:
    """
    Nome do modelo no MLflow Model Registry.
    Formato: <team>.<domain>.<agent_name>
    Exemplo: meu-time.geral.tutorial-agente
    """
    return f"{_team_name()}.{_domain()}.{_agent_name()}"


def registration_experiment_name() -> str:
    """Experimento dedicado ao registro/promoção do modelo."""
    return f"{_team_name()}/{_domain()}/{_agent_name()}/model-registration"


def dev_experiment_name() -> str:
    """Experimento de desenvolvimento onde as runs do agente ficam."""
    return f"{_team_name()}/{_domain()}/{_agent_name()}"


def _get_latest_dev_run() -> pd.Series:
    """
    Busca a run mais recente do experimento de desenvolvimento.
    Usada para rastrear de qual run dev o modelo foi promovido.
    """
    dev_experiment = mlflow.get_experiment_by_name(dev_experiment_name())

    if dev_experiment is None:
        raise ValueError(
            f"Experimento dev não encontrado: {dev_experiment_name()}\n"
            "Execute agents/agent.py ou agents/agent_mock.py pelo menos uma vez."
        )

    runs = mlflow.search_runs(
        experiment_ids=[dev_experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(
            f"Nenhuma run encontrada no experimento dev: {dev_experiment_name()}"
        )

    return runs.iloc[0]


# ── Função principal ──────────────────────────────────────────────────────────

def register_model(agent_file: str) -> mlflow.models.model.ModelInfo:
    """
    Empacota o agente como MLflow PyFunc e registra no Model Registry.

    Fluxo:
      1. Valida e resolve o caminho absoluto do arquivo do agente
      2. Busca a última run dev para rastreabilidade
      3. Abre uma run de registro no experimento dedicado
      4. Loga o modelo com AgentPyfuncWrapper + artefatos + assinatura
      5. Registra no Model Registry com o nome <team>.<domain>.<agent>

    Args:
        agent_file: Caminho para o arquivo do agente (obrigatório).
                    Exemplo: "agents/agent.py"

    Returns:
        ModelInfo com model_uri e run_id do modelo registrado.
    """
    # ── Valida o caminho do agente ────────────────────────────────────────────
    if not agent_file:
        raise ValueError(
            "O caminho do arquivo do agente é obrigatório.\n"
            "Exemplo: register_model(agent_file='agents/agent.py')"
        )

    agent_file = str(Path(agent_file).resolve())

    if not Path(agent_file).exists():
        raise FileNotFoundError(f"Arquivo do agente não encontrado: {agent_file}")

    # ── Configura MLflow ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(registration_experiment_name())

    # ── Busca a última run dev para rastreabilidade ───────────────────────────
    latest_dev_run = _get_latest_dev_run()

    # ── Define exemplos de input/output para a assinatura do modelo ──────────
    input_example = pd.DataFrame([{"message": "Qual a capital do Brasil?"}])
    output_example = pd.DataFrame([{"answer": "Brasília é a capital do Brasil."}])

    signature = infer_signature(
        model_input=input_example,
        model_output=output_example,
    )

    model_name = registered_model_name()

    # ── Abre a run de registro ────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"{_agent_name()}-registration") as run:

        # Tags de governança e rastreabilidade
        mlflow.set_tags(
            {
                "ai_platform.phase": "model_registration",
                "ai_platform.framework": "langchain",
                "ai_platform.agent_name": _agent_name(),
                "ai_platform.team": _team_name(),
                "ai_platform.domain": _domain(),
                "ai_platform.model_logged": "true",
                "ai_platform.promoted_from_environment": "dev",
                "ai_platform.promoted_from_run_id": latest_dev_run["run_id"],
            }
        )

        # Parâmetros para auditoria
        mlflow.log_param("agent_name", _agent_name())
        mlflow.log_param("agent_file", agent_file)
        mlflow.log_param("registered_model_name", model_name)
        mlflow.log_param("source_dev_run_id", latest_dev_run["run_id"])
        mlflow.log_param("source_dev_experiment_id", latest_dev_run["experiment_id"])

        # ── Loga e registra o modelo ──────────────────────────────────────────
        model_info = mlflow.pyfunc.log_model(
            artifact_path=_artifact_path(),
            python_model=AgentPyfuncWrapper(),
            artifacts={
                # O MLflow copia este arquivo para dentro do artefato do modelo,
                # garantindo que o agente seja portável e versionado junto ao modelo.
                "agent_file": agent_file
            },
            registered_model_name=model_name,
            input_example=input_example,
            signature=signature,
            pip_requirements=[
                "mlflow>=3.0,<4.0",
                "pandas",
                "langchain>=0.3",
                "langchain-core>=0.3",
                "langchain-openai>=1.2.2",
                "langchain-community>=0.3",
                "langchain-tavily>=0.1",
                "python-dotenv>=1.0",
            ],
        )

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 56)
    print("  ✅ MODELO REGISTRADO COM SUCESSO")
    print("═" * 56)
    print(f"  Agent name         : {_agent_name()}")
    print(f"  Registered model   : {model_name}")
    print(f"  Registration run   : {run.info.run_id}")
    print(f"  Source dev run     : {latest_dev_run['run_id']}")
    print(f"  Model URI          : {model_info.model_uri}")
    print(f"  MLflow UI          : {_tracking_uri()}")
    print("═" * 56 + "\n")

    return model_info


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empacota e registra o agente como modelo MLflow PyFunc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  uv run python model/register_model.py
  uv run python model/register_model.py --agent-file agents/agent.py
        """,
    )
    parser.add_argument(
        "--agent-file",
        default=None,
        help="Caminho para o arquivo do agente (padrão: agents/agent.py).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    register_model(agent_file=args.agent_file)
