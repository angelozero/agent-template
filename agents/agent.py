import os
import sys
from pathlib import Path

# Garante que a raiz do projeto está no sys.path independente de como o arquivo
# é executado (python agents/agent.py ou uv run python agents/agent.py).
# Deve ficar ANTES de qualquer import de módulos do projeto (config, etc).
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mlflow
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from config import track_agent

load_dotenv()

"""
Agente com LLM real via LiteLLM Proxy.
"""

def build_model():
    """Constrói o modelo LLM apontando para o LiteLLM Proxy."""
    model = init_chat_model(
        model="gpt-4o-mini",
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
    # Garante que a raiz do projeto está no sys.path quando o arquivo
    # é executado diretamente (python agents/agent.py).
    # Sem isso, `from config import ...` falha pois o Python usa agents/ como base.
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    message = (
        sys.argv[1]
        if len(sys.argv) > 1
       else "Quantas capitais tem o Brasil?"
       # else "Quantas capitais tem o Brasil em relação ao Japão?"
       # else "Nas minhas viagens eu estive no oriente, em relação ao Brasil quantos continentes no oriente nós temos?"
       # else "Quantas ilhas nós temos em volta do Brasil?"
       # else "A bandeira do Brasil tem estrelas que se não me enganam representam as capitais do Brasil, é igual as estrelas da bandeira dos Estados Unidos?"
    )
    result = invoke_agent(message)
    print(f"\n{result}\n")
