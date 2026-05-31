"""
Configuração dos LLM Judges para avaliação via MLflow.

Responsabilidades:
1. Mapeia LLM_API_KEY → OPENAI_API_KEY (autenticação dos judges)
2. Redireciona os judges para o LiteLLM Proxy (em vez de api.openai.com)
3. Define o modelo padrão dos judges

Uso:
    from config.judge_config import JUDGE_MODEL, setup_judge_provider
    setup_judge_provider()
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── Modelo usado pelos LLM Judges ────────────────────────────────────────────
# Deve estar disponível no LiteLLM Proxy. Formato: "openai:/<model_name>"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai:/gpt-5")


def setup_judge_provider():
    """
    Configura o ambiente para que os LLM Judges do MLflow usem o LiteLLM Proxy.

    Os scorers (Correctness, RelevanceToQuery, Guidelines) usam a API OpenAI
    internamente. Esta função:
    - Mapeia LLM_API_KEY → OPENAI_API_KEY
    - Aplica um patch no MLflow para redirecionar as chamadas do provider OpenAI
      para o LLM_BASE_URL (LiteLLM Proxy), já que o MLflow hardcodes
      "https://api.openai.com/v1" como endpoint padrão.
    """
    _setup_openai_api_key()
    _patch_openai_provider()


def _setup_openai_api_key():
    """Mapeia LLM_API_KEY → OPENAI_API_KEY se não estiver definida."""
    if os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("LLM_API_KEY")


def _patch_openai_provider():
    """
    Patch no MLflow para redirecionar o provider OpenAI para o LiteLLM Proxy.

    O MLflow cria o OpenAIConfig sem openai_api_base, fazendo os judges chamarem
    api.openai.com diretamente. Este patch intercepta a criação do provider e
    injeta o LLM_BASE_URL como openai_api_base.
    """
    llm_base_url = os.getenv("LLM_BASE_URL")
    if not llm_base_url:
        return

    import mlflow.metrics.genai.model_utils as _model_utils
    from mlflow.gateway.config import EndpointConfig, OpenAIConfig, Provider
    from mlflow.gateway.providers.openai import OpenAIProvider

    _original_get_provider = _model_utils._get_provider_instance

    def _patched_get_provider(provider, model, base_url=None):
        """Injeta LLM_BASE_URL como openai_api_base para o provider OpenAI."""
        if provider == Provider.OPENAI:
            config = OpenAIConfig(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                openai_api_base=llm_base_url.rstrip("/"),
            )
            route_config = EndpointConfig(
                name=provider,
                endpoint_type="llm/v1/chat",
                model={
                    "provider": provider,
                    "name": model,
                    "config": config.model_dump(),
                },
            )
            return OpenAIProvider(route_config)
        return _original_get_provider(provider, model, base_url=base_url)

    _model_utils._get_provider_instance = _patched_get_provider
