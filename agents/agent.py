import os
import sys
import mlflow
from dotenv import load_dotenv
from config import track_agent

load_dotenv()

"""
Agente com LLM real via LiteLLM Proxy.
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from config import track_agent

load_dotenv()

def build_model():
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
    model = build_model()
    
    ### Carrega o prompt do MLflow Registry
    # "prompts:/prompt_cidades_capitais_br@latest" = última versão
    # "prompts:/prompt_cidades_capitais_br/1"      = versão específica
    # "prompts:/prompt_cidades_capitais_br@prod"   = alias (ex: produção)
    prompt = mlflow.genai.load_prompt("prompts:/prompt_cidades_capitais_br@latest")

    # Preenche as variáveis do template
    domain = os.getenv("DOMAIN", "domain_bar")
    messages = prompt.format(domain=domain, user_message=message)
    
    # Chamada simples: envia mensagem e recebe resposta
    response = model.invoke(messages)
    return {"role": "assistant", "content": response.content}


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "Qual a capital do Brasil?"
    result = invoke_agent(message)
    print(result)
