import os
from dataclasses import dataclass


# @dataclasse ---> Cria um objeto imutável que ninguém pode alterar.
@dataclass(frozen=True)
class Config:
    agent_name: str
    team: str
    domain: str
    enviroment: str
    mlflow_tracking_uri: str
    llm_base_url: str
    llm_api_key: str

def load_config() -> Config:
    agent_name = os.environ.get("AGENT_NAME")
    if not agent_name:
        raise EnvironmentError("AGENT_NAME value was not found in .env file")

    llm_api_key = os.environ.get("LLM_API_KEY")
    if not llm_api_key or llm_api_key == "define":
        raise EnvironmentError("LLM_API_KEY value was not found in .env file")

    return Config(
        agent_name=agent_name,
        team=os.getenv("TEAM_NAME", "default"),
        domain=os.getenv("DOMAIN", "geral"),
        enviroment=os.getenv("ENVIROMENT", "dev"),
        mlflow_tracking_uri=os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5050"
        ),
        llm_base_url=os.getenv(
            "LLM_BASE_URL", "YOUR_LLM_URL"
        ),
        llm_api_key=llm_api_key,
    )


# Is there a difference between "==" and "is"?
# is will return True if two variables point to the same object (in memory), == if the objects referred to by the variables are equal
# https://stackoverflow.com/questions/132988/is-there-a-difference-between-and-is
