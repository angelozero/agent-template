"""
Baixa (carrega) um modelo registrado no MLflow Model Registry.

Uso:
    uv run python model/load_model.py
    uv run python model/load_model.py --model-uri "models:/meu-time.geral.tutorial-agente/1"
"""

import argparse
import os
import sys
from pathlib import Path

# Garante que a raiz do projeto está no sys.path independente de como o arquivo
# é executado (python model/load_model.py ou uv run python model/load_model.py).
# Deve ficar ANTES de qualquer import de módulos do projeto (config, etc).
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mlflow
from dotenv import load_dotenv

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


def default_model_uri() -> str:
    """
    URI padrão para a última versão do modelo registrado.
    Formato: models:/<team>.<domain>.<agent>/latest
    """
    model_name = f"{_team_name()}.{_domain()}.{_agent_name()}"
    return f"models:/{model_name}/latest"


# ── Função principal ──────────────────────────────────────────────────────────

def load_model(model_uri: str | None = None) -> mlflow.pyfunc.PyFuncModel:
    """
    Baixa e carrega um modelo PyFunc do MLflow Model Registry.

    Ao carregar, o MLflow chama load_context() do AgentPyfuncWrapper,
    que importa dinamicamente o agents/agent.py armazenado como artefato.

    Args:
        model_uri: URI do modelo. Formatos aceitos:
                   - "models:/<registered_model_name>/latest"  → última versão
                   - "models:/<registered_model_name>/1"       → versão específica
                   - "runs:/<run_id>/agent"                    → direto de uma run
                   Se None, usa models:/<team>.<domain>.<agent>/latest

    Returns:
        Instância de mlflow.pyfunc.PyFuncModel pronta para chamar .predict().
    """
    mlflow.set_tracking_uri(_tracking_uri())

    if model_uri is None:
        model_uri = default_model_uri()

    print("\n" + "═" * 56)
    print("  🔽 CARREGANDO MODELO DO MLFLOW")
    print("═" * 56)
    print(f"  Model URI  : {model_uri}")
    print(f"  MLflow UI  : {_tracking_uri()}")
    print("─" * 56)

    # mlflow.pyfunc.load_model() resolve o artefato, copia os arquivos
    # para um diretório temporário e chama load_context() do wrapper.
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    print("  ✅ Modelo carregado com sucesso.")
    print("═" * 56 + "\n")

    return loaded_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa um modelo registrado no MLflow Model Registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Carregar a última versão (usa AGENT_NAME, TEAM_NAME e DOMAIN do .env)
  uv run python model/load_model.py

  # Carregar uma versão específica
  uv run python model/load_model.py --model-uri "models:/meu-time.geral.tutorial-agente/1"

  # Carregar direto de uma run
  uv run python model/load_model.py --model-uri "runs:/<run_id>/agent"
        """,
    )
    parser.add_argument(
        "--model-uri",
        default=None,
        help=(
            "URI do modelo no MLflow. "
            "Padrão: models:/<team>.<domain>.<agent>/latest"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    load_model(model_uri=args.model_uri)
