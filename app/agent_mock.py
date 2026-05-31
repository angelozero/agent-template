import os
from dotenv import load_dotenv
from ai_platform import track_agent

load_dotenv()

### Mock LLM
# Em vez de chamar o LLM Gateway real, retornamos respostas fixas.
# Isso permite testar toda a integração MLflow sem credenciais.


class MockLLM:
    """Simula um LLM que responde com texto fixo."""

    def invoke(self, messages):
        user_msg = (
            messages[-1]["content"]
            if isinstance(messages[-1], dict)
            else str(messages[-1])
        )
        return {
            "role": "assistant",
            "content": f"[MOCK] Recebi sua mensagem: '{user_msg}'. "
            f"Esta é uma resposta simulada do LLM.",
            "model": "mock-gpt-4.1",
            "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35},
        }


def build_agent():
    """Constrói o 'agente' — neste caso, apenas o mock."""
    return MockLLM()


@track_agent
def invoke_agent(message: str):
    """Ponto de entrada do agente. Mantenha essa assinatura"""
    agent = build_agent()
    result = agent.invoke([{"role": "user", "content": message}])
    return result


if __name__ == "__main__":
    import sys

    message = (
        sys.argv[1] if len(sys.argv) > 1 else "Qual a previsão do tempo em São Paulo?"
    )
    invoke_agent(message)

# O que @track_agent (ai_platform/tracking.py) faz ?
# @track_agent
# def invoke_agent(message: str):
#     """Só executa a lógica do agente. Nada mais."""
#     result = f"Resposta para: {message}"
#     return result
# -------------------------------------------------------
# -------------------------------------------------------
# invoke_agent("Olá")
# O que realmente acontece quando você chama invoke_agent("Olá") com o decorator:
# ┌─────────────────────────────────────────────────────┐
# │  ANTES (código injetado pelo @track_agent)          │
# │                                                     │
# │  1. load_dotenv()                                   │
# │  2. cfg = load_config()                             │
# │  3. mlflow.set_tracking_uri("http://localhost:5000")│
# │  4. mlflow.set_experiment("/time/dominio/agente")   │
# │  5. mlflow.langchain.autolog(log_traces=True)       │
# │  6. mlflow.start_run("meu-agente-dev")              │
# │  7. mlflow.set_tags({team, domain, ...})            │
# │  8. mlflow.log_text("Olá", "input/user_message.txt")│
# ├─────────────────────────────────────────────────────┤
# │  EXECUÇÃO ORIGINAL (seu código, intocado)           │
# │                                                     │
# │  result = f"Resposta para: Olá"                     │
# │  return result                                      │
# ├─────────────────────────────────────────────────────┤
# │  DEPOIS (código injetado pelo @track_agent)         │
# │                                                     │
# │  9. mlflow.log_text(result, "output/...")           │
# │  10. print("Run ID: abc123...")                     │
# │  11. mlflow.end_run()                               │
# └─────────────────────────────────────────────────────┘