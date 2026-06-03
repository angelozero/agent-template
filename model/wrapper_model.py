"""
Wrapper PyFunc para empacotar o agente (agents/agent.py) como modelo MLflow.

O MLflow exige que modelos customizados implementem mlflow.pyfunc.PythonModel.
Esta classe:
  1. Carrega dinamicamente o módulo do agente via importlib (load_context)
  2. Executa o agente e retorna a resposta em formato DataFrame (predict)

Uso:
    Instancie AgentPyfuncWrapper() e passe para mlflow.pyfunc.log_model().
"""

import importlib.util

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel


class AgentPyfuncWrapper(PythonModel):
    """
    Empacota agents/agent.py como um modelo MLflow PyFunc.

    O MLflow serializa esta classe junto com os artefatos declarados em
    `artifacts={"agent_file": <caminho absoluto>}`. Na hora de carregar
    (load_context), o arquivo do agente é importado dinamicamente para que
    nenhuma dependência de path seja hardcoded.
    """

    # ------------------------------------------------------------------
    # load_context — chamado UMA VEZ quando o modelo é carregado
    # ------------------------------------------------------------------
    def load_context(self, context):
        """
        Carrega o módulo do agente a partir do artefato 'agent_file'.

        O MLflow resolve o caminho real do arquivo e o disponibiliza em
        context.artifacts["agent_file"]. Usamos importlib para importar
        o módulo sem precisar que ele esteja no sys.path.
        """
        agent_file = context.artifacts["agent_file"]

        spec = importlib.util.spec_from_file_location("user_agent", agent_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Guarda o módulo para uso em predict()
        self.agent_module = module

    # ------------------------------------------------------------------
    # predict — chamado a cada inferência
    # ------------------------------------------------------------------
    def predict(self, context, model_input):
        """
        Executa o agente e retorna a resposta como DataFrame.

        Aceita dois formatos de entrada:
          - pd.DataFrame com coluna "message"
          - dict com chave "message"

        Retorna:
          pd.DataFrame com coluna "answer" contendo a resposta do agente.
        """
        # ── Extrai a mensagem do input ────────────────────────────────
        if isinstance(model_input, pd.DataFrame):
            message = model_input["message"].iloc[0]

        elif isinstance(model_input, dict):
            message = model_input["message"]

        else:
            raise ValueError(
                "model_input deve ser pd.DataFrame ou dict com campo 'message'. "
                f"Recebido: {type(model_input)}"
            )

        # ── Chama o agente ────────────────────────────────────────────
        # invoke_agent já possui o @track_agent decorator, que abre um
        # mlflow.start_run() internamente. Por isso NÃO abrimos outro run aqui
        # para evitar runs aninhados desnecessários.
        result = self.agent_module.invoke_agent(message)

        # ── Normaliza o resultado para DataFrame ──────────────────────
        if isinstance(result, dict):
            answer = result.get("content", str(result))
        else:
            answer = str(result)

        return pd.DataFrame([{"answer": answer}])
