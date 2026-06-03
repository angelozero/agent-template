import mlflow
import functools
from dotenv import load_dotenv
from mlflow.entities import ViewType
from config.app_config import load_config
from typing import Any, Callable, TypeVar

# TypeVar("F") é para type hints — diz ao mypy que o tipo de retorno é o mesmo tipo da função decorada.
F = TypeVar("F", bound=Callable[..., Any])


# track_agent recebe uma função (func) e retorna uma nova função (wrapper) que "envolve" a original.
def track_agent(func: F) -> F:
    """Decorator que registra logs de execução no MLflow."""

    # @functools.wraps(func) preserva o nome e docstring da função original.
    # Quando usamos @functools.wraps(func), o Python guarda a função original em wrapper.__wrapped__.
    # O interceptor usa isso para chamar a função sem o decorator
    # Isso evita que mlflow.start_run() seja chamado duas vezes (uma pelo interceptor, outra pelo decorator).
    @functools.wraps(func)
    def wrapper(message: str, **kwargs: Any) -> Any:
        load_dotenv()
        cfg = load_config()
        _setup_mlflow(cfg)
        experiment = _experiment_name(cfg)

        # Abre uma 'run' - uma execução indivídual
        with mlflow.start_run(run_name=f"{cfg.agent_name}-{cfg.enviroment}") as run:
            # Tags de governança - metadados para filtrar/buscar depois
            mlflow.set_tags(
                {
                    "ai_platform.agent_name": cfg.agent_name,
                    # "ai_platform.team": cfg.team,
                    "ai_platform.domain": cfg.domain,
                    "ai_platform.enviroment": cfg.enviroment,
                    "ai_platform.framework": "langchain",
                    "ai_platform.is_agent": "true",
                }
            )

            # Salva input como artifact
            mlflow.log_text(message, "input/user_message.txt")

            # Executa a função do agente
            result = func(message, **kwargs)

            # Salva o output como artifact
            _log_output(result)
            _print_run_summary(run.info.run_id, experiment, cfg.mlflow_tracking_uri)

        return result

    return wrapper


def _log_output(result: Any) -> None:
    if isinstance(result, dict):
        mlflow.log_dict(result, "output/final_response.json")
    else:
        mlflow.log_text(str(result), "output/final_response.txt")


def _print_run_summary(run_id: str, experiment: str, tracking_uri: str) -> None:
    print(f"\n{'─' * 52}")
    print(f"  MLflow Run ID  : {run_id}")
    print(f"  Experimento    : {experiment}")
    print(f"  UI             : {tracking_uri}")
    print(f"{'─' * 52}\n")


### MLFLOW ###
def _setup_mlflow(cfg) -> None:
    """Configura o MLflow: URI, experimento e autolog"""
    # set_tracking_uri: diz ao MLflow onde está o servidor (Docker local).
    experiment_name = _experiment_name(cfg)
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment and experiment.lifecycle_stage == "deleted":
        # Experimento existe mas foi deletado — restaura para poder reutilizá-lo
        mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)

    # set_experiment cria o experimento se não existir, ou seleciona se já existir (ativo)
    mlflow.set_experiment(experiment_name)

    # langchain.autolog(log_traces=True): esta é a linha mais importante.
    # Ela faz o MLflow "espionar" todas as chamadas LangChain e registrar traces automaticamente.
    # Sem este trecho os logs teriam que serem feitos manualmente a cada chamada a LLM.
    mlflow.langchain.autolog(log_traces=True)


# set_experiment: cria/seleciona o experimento com nome hierárquico /<TIME>/<DOMINIO>/<AGENTE>.
def _experiment_name(cfg) -> str:
    """Gera o nome do experimento: /<TIME>/<DOMINIO>/<AGENTE>"""
    return f"{cfg.team}/{cfg.domain}/{cfg.agent_name}"
