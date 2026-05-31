import os
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

prompt_v1 = mlflow.genai.register_prompt(
    name="prompt_cidades_capitais_br",
    template=[
        {
            "role": "system",
            "content": (
                "Você é um assistente com conhecimento sobre o Brasil"
                "Responda de forma clara, concisa e profissional"
                "Domínio: {{domain}}."
                "Se não souber a resposta, diga que não sabe"
            ),
        },
        {"role": "user", "content": "{{user_message}}"},
    ],
    commit_message="V1: Prompt Capitas Brasileiras",
    tags={"author": "tutorial", "domain": "geral"},
)

print(f"\n✅ Prompt registrado com sucesso - prompt: {prompt_v1.name} - version: {prompt_v1.version}\n")