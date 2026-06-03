import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel

load_dotenv()


class AgentPyfuncWrapper(PythonModel):

    def load_context(self, context):
        import importlib.util

        agent_file = context.artifacts["agent_file"]

        spec = importlib.util.spec_from_file_location(
            "user_agent",
            agent_file
        )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.agent_module = module

    def predict(self, context, model_input):

        prod_experiment_name = get_experiment_name(
            agent_name="brandaoagent1119", # parametrizar
            environment_name="prod"
        )

        mlflow.set_experiment(prod_experiment_name)

        mlflow.langchain.autolog()

        with mlflow.start_run( # isso eh oq cria o evaluation run tradicional e nao o trace
                run_name="brandaoagent1119-prod" # parametrizar
        ) as run:

            if isinstance(model_input, pd.DataFrame):
                message = model_input["message"].iloc[0]

            elif isinstance(model_input, dict):
                message = model_input["message"]

            else:
                raise ValueError(
                    "model_input precisa conter campo 'message'"
                )

            result = self.agent_module.invoke_agent(message)

            return pd.DataFrame([{
                "answer": str(result)
            }])


def get_experiment_name(
    agent_name: str,
    environment_name: str = "dev",
) -> str:

    return (
        f"/Users/"
        f"{os.getenv('DATABRICKS_USER', 'gbrandao@ciandt.com')}"
        f"/ai-platform/agents/{agent_name}-{environment_name}"
    )


def get_latest_dev_run(agent_name: str):

    dev_experiment_name = get_experiment_name(
        agent_name=f"{agent_name}",
        environment_name="dev"
    )

    dev_experiment = mlflow.get_experiment_by_name(
        dev_experiment_name
    )

    if dev_experiment is None:
        raise ValueError(
            f"Experimento dev nao encontrado: {dev_experiment_name}"
        )

    runs = mlflow.search_runs(
        experiment_ids=[dev_experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError(
            f"Nenhuma run encontrada no experimento dev: {dev_experiment_name}"
        )

    return runs.iloc[0]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--agent-file", required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--registered-model-name", required=True)

    args = parser.parse_args()

    agent_file = str(Path(args.agent_file).resolve())

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "databricks")
    )

    latest_dev_run = get_latest_dev_run(
        agent_name=args.agent_name
    )

    prod_experiment_name = get_experiment_name(
        agent_name=args.agent_name,
        environment_name = "prod"
    )

    mlflow.set_experiment(prod_experiment_name)

    input_example = pd.DataFrame([{
        "message": "Qual a previsao do tempo em maceio?"
    }])

    output_example = pd.DataFrame([{
        "answer": "Tempo quente e sem chuva em Maceio"
    }])

    signature = infer_signature(
        model_input=input_example,
        model_output=output_example
    )

    with mlflow.start_run(
        run_name=f"{args.agent_name}-registration"
    ) as run:

        mlflow.set_tags({
            "ai_platform.phase": "model_registration",
            "ai_platform.framework": "langchain",
            "ai_platform.agent_name": args.agent_name,
            "ai_platform.environment": "prod",
            "ai_platform.model_logged": "true",
            "ai_platform.promoted_from_environment": "dev",
            "ai_platform.promoted_from_run_id": latest_dev_run["run_id"],
        })

        mlflow.log_param("agent_name", args.agent_name)
        mlflow.log_param("agent_file", agent_file)
        mlflow.log_param("registered_model_name", args.registered_model_name)
        mlflow.log_param("source_dev_run_id", latest_dev_run["run_id"])
        mlflow.log_param("source_dev_experiment_id", latest_dev_run["experiment_id"])

        model_info = mlflow.pyfunc.log_model(
            artifact_path="agent",
            python_model=AgentPyfuncWrapper(),
            artifacts={
                "agent_file": agent_file
            },
            registered_model_name=args.registered_model_name,
            input_example=input_example,
            signature=signature,
            pip_requirements=[
                "mlflow",
                "pandas",
                "langchain",
                "langchain-openai",
                "langchain-tavily",
                "python-dotenv",
                "tavily-python"
            ]
        )

        print("\n=== MODELO REGISTRADO ===")
        print(f"prod_run_id: {run.info.run_id}")
        print(f"source_dev_run_id: {latest_dev_run['run_id']}")
        print(f"prod_experiment: {prod_experiment_name}")
        print(f"model_uri: {model_info.model_uri}")


if __name__ == "__main__":
    main()


"""
python ai_platform_register_model.py \
  --agent-file ./agent.py \
  --agent-name brandaoagent1119 \
  --registered-model-name alberteinstein.labgb.brandaoagent1119-prod
"""