import os
from dotenv import load_dotenv
from ai_platform import track_agent

load_dotenv()

"""
Agente com LLM real via LiteLLM Proxy.
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from ai_platform import track_agent

load_dotenv()

SYSTEM_PROMPT = (
    """Você é um assistente prestativo. Responda de forma clara e concisa."""
)


def build_agent():
    """Constrói o modelo LLM apontando para o LiteLLM Proxy."""
    model = init_chat_model(
        model="gpt-5",
        model_provider="openai",
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )
    return model


@track_agent
def invoke_agent(message: str):
    """Ponto de entrada do agente."""
    model = build_agent()
    # Chamada simples: envia mensagem e recebe resposta
    response = model.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
    )
    return {"role": "assistant", "content": response.content}


if __name__ == "__main__":
    import sys

    message = sys.argv[1] if len(sys.argv) > 1 else "Qual a capital do Brasil?"
    result = invoke_agent(message)
    print(result)
