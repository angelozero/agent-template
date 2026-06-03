"""
Wrapper PyFunc para empacotar o agente (agents/agent.py) como modelo MLflow.

O MLflow exige que modelos customizados implementem mlflow.pyfunc.PythonModel.
Esta classe:
  1. Carrega dinamicamente o módulo do agente via importlib (load_context)
  2. Ativa mlflow.langchain.autolog() ANTES de qualquer run (load_context)
  3. Executa o agente SEM o @track_agent para evitar run aninhada (predict)
  4. Retorna a resposta em formato DataFrame (predict)

Por que não usar @track_agent diretamente:
  Quando o modelo é servido pelo MLflow (load_model + predict), o MLflow já
  gerencia um contexto de run internamente. Se @track_agent abrir outra run
  (nested), o mlflow.langchain.autolog() não captura os traces corretamente.
  A solução é chamar a função original via __wrapped__ (guardada pelo
  functools.wraps do decorator) e deixar o autolog capturar os traces
  dentro do contexto de run já existente.

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
        Carrega o módulo do agente e ativa o autolog de traces.

        CORREÇÃO DE TRACE: mlflow.langchain.autolog() deve ser chamado
        ANTES de qualquer run ser aberta. Chamá-lo aqui garante que o
        interceptor LangChain esteja ativo quando predict() for executado,
        capturando corretamente os traces das chamadas LLM.

        O MLflow resolve o caminho real do arquivo e o disponibiliza em
        context.artifacts["agent_file"]. Usamos importlib para importar
        o módulo sem precisar que ele esteja no sys.path.
        """
        # ── Ativa autolog ANTES de qualquer run ──────────────────────
        # Isso registra o interceptor LangChain globalmente no processo,
        # garantindo que as chamadas LLM sejam capturadas como traces.
        mlflow.langchain.autolog(log_traces=True)

        # ── Carrega o módulo do agente dinamicamente ──────────────────
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

        CORREÇÃO DE TRACE: chama invoke_agent.__wrapped__ (a função original
        sem o @track_agent) para evitar que mlflow.start_run() seja aberto
        dentro de um contexto de run já existente (gerenciado pelo MLflow
        ao servir o modelo). O autolog ativado em load_context() captura
        os traces automaticamente dentro do contexto correto.

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

        # ── Chama o agente SEM o @track_agent ────────────────────────
        # invoke_agent.__wrapped__ é a função original guardada pelo
        # functools.wraps() usado no decorator @track_agent.
        # Isso evita que mlflow.start_run() seja chamado dentro de uma
        # run já aberta pelo MLflow, o que impediria a captura de traces.
        invoke_fn = getattr(
            self.agent_module.invoke_agent,
            "__wrapped__",
            self.agent_module.invoke_agent,  # fallback: usa com decorator se __wrapped__ não existir
        )
        result = invoke_fn(message)

        # ── Normaliza o resultado para DataFrame ──────────────────────
        if isinstance(result, dict):
            answer = result.get("content", str(result))
        else:
            answer = str(result)

        return pd.DataFrame([{"answer": answer}])
