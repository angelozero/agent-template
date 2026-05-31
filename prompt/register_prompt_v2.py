import os
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

prompt_v2 = mlflow.genai.register_prompt(
    # se 'name' estiver com o mesmo valor de prompt_v1 ele será sobrescrito mas manterá os versionamentos
    name="prompt_cidades_capitais_br",
    template=[
        {
            "role": "system",
            "content": (
                "Você é um assistente especializado em capitais e cidades brasileiras. "
                "Domínio: {{domain}}. "
                "REGRAS: "
                "1. Responda SEMPRE em português brasileiro. "
                "2. Seja conciso (máximo 3 parágrafos). "
                "3. Se a pergunta for sobre qualquer outro assunto que não seja capitais ou cidades brasileiras, inclua o disclaimer: "
                "'Não tenho conhecimento sobre o assunto.' "
                "4. Se não souber, diga 'Não tenho essa informação.'"
            ),
        },
        {
            "role": "user",
            "content": "{{user_message}}",
        },
    ],
    commit_message="V2: Prompt Capitas Brasileiras",
    tags={
        "author": "tutorial",
        "domain": "geral",
        "change": "added-format-rules",
    },
)

print(f"\n✅ Prompt registrado com sucesso - prompt: {prompt_v2.name} - version: {prompt_v2.version}\n")
