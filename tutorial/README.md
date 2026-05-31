# 🎓 Tutorial: Construindo a Stack de Referência para Agentes do Zero

Template de projeto para desenvolvimento local de agentes de IA generativa integrado ao **MLflow GenAI** e ao **LLM Gateway corporativo**.

---

## Índice das Etapas

| Etapa | Tema | O que você aprende |
|---|---|---|
| [Etapa 1](ETAPA_01.md) | Estrutura do Projeto | Diretórios, `pyproject.toml`, conceitos-chave |
| [Etapa 2](ETAPA_02.md) | Configuração | `.env`, dataclass imutável, validação fail-fast |
| [Etapa 3](ETAPA_03.md) | `@track_agent` + MLflow | Decorator pattern, API MLflow, autolog |
| [Etapa 4](ETAPA_04.md) | Docker + Justfile | MLflow local, backend store, automação |
| [Etapa 5](ETAPA_05.md) | Agente com Mock | Testar integração MLflow sem credenciais |
| [Etapa 6](ETAPA_06.md) | LLM Real (LiteLLM) | `init_chat_model`, traces reais, troubleshooting |
| [Etapa 7](ETAPA_07.md) | Prompt Registry | Versionamento de prompts, `load_prompt` |
| [Etapa 8](ETAPA_08.md) | Evaluation + Judges | Dataset de teste, Correctness, Guidelines |

---

## Pré-requisitos

| Ferramenta | Versão mínima | Instalação |
|---|---|---|
| Python | 3.12 | [python.org](https://python.org) |
| uv | qualquer | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker | qualquer | [docker.com](https://docker.com) |
| just | qualquer | `brew install just` / `cargo install just` |

---

## Início rápido

```bash
# 1. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env: defina AGENT_NAME, TEAM_NAME, DOMAIN e LLM_API_KEY

# 2. Instale as dependências
just install

# 3. Suba o MLflow local
just up

# 4. Execute o agente
python examples/agent.py "Qual a capital do Brasil?"
```

Abra **http://localhost:5000** para visualizar o experimento registrado.

---

## Mapa evolutivo

```
Etapa 1 — Estrutura + pyproject.toml
    ↓
Etapa 2 — Config (.env + dataclass)
    ↓
Etapa 3 — @track_agent + MLflow (o coração)
    ↓
Etapa 4 — Docker MLflow + Justfile
    ↓
Etapa 5 — Agente com Mock
    ↓
Etapa 6 — LLM Real (LiteLLM Proxy)
    ↓
Etapa 7 — Prompt Registry (versionamento)
    ↓
Etapa 8 — Evaluation + Judges
```

Comece pela [Etapa 1](ETAPA_01.md) →
