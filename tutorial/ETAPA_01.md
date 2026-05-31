# Etapa 1 — Estrutura do Projeto e Configuração

## Visão Geral da Arquitetura

Antes de codar, vamos entender **o que** estamos construindo e **por quê**.

Este projeto é um **template** que resolve um problema simples: permitir que desenvolvedores criem agentes de IA e automaticamente tenham **rastreabilidade completa** via MLflow — sem precisar entender MLflow.

### O que o desenvolvedor ganha ao usar este template:

- Experimentos organizados por time e domínio no MLflow
- Traces completos de cada chamada LLM
- Histórico de execuções para comparar versões e depurar comportamentos

### Conceitos-chave

| Conceito | O que é | Analogia |
|---|---|---|
| **Experiment** | Agrupamento lógico de execuções | Uma "pasta" no MLflow: `/<time>/<domínio>/<agente>` |
| **Run** | Uma execução individual do agente | Um "registro" com input, output, métricas |
| **Trace** | Detalhamento de cada chamada LLM dentro de um Run | Um "raio-X" mostrando prompts, respostas, tokens |
| **Autolog** | MLflow intercepta automaticamente chamadas LangChain | "Espião" que captura tudo sem código extra |

---

## O que vamos aprender nesta etapa

- Criar a estrutura de diretórios
- Configurar `pyproject.toml` com dependências
- Entender o papel de cada arquivo

---

## Passo 1.1 — Criar a estrutura de diretórios

```bash
mkdir -p meu-agente/{config,agents,prompt,dataset,judge,docker,tutorial}
cd meu-agente
```

Estrutura alvo:

```
meu-agente/
├── config/               # Configuração e tracking MLflow
│   ├── __init__.py
│   ├── app_config.py
│   ├── judge_config.py
│   └── tracking.py
│
├── agents/
│   ├── agent_mock.py     # agente com LLM mockado
│   └── agent.py          # agente com LLM real
│
├── prompt/
│   ├── register_prompt_v1.py
│   ├── register_prompt_v2.py
│   ├── register_prompt_v3.py
│   └── register_prompt_v4.py
│
├── dataset/
│   └── register_dataset.py
│
├── judge/
│   └── register_judge.py
│
├── docker/
│   └── docker-compose.yml
│
├── tutorial/             # este tutorial
│
├── .env_example
├── .gitignore
└── pyproject.toml
```

---

## Passo 1.2 — Criar o `pyproject.toml`

```toml
[project]
name = "hiae-ai-agent-template"
version = "0.1.0"
description = "Template de referência para agentes de IA generativa — HIAE AI Platform"
requires-python = ">=3.12"
dependencies = [
    # Plataforma AI (não remover)
    "mlflow[langchain]>=3.0,<4.0",
    "python-dotenv>=1.0",
    # Framework de agentes (adapte conforme necessário)
    "langchain>=0.3",
    "langchain-core>=0.3",
    "langgraph>=0.2",
    "langchain-community>=0.3",
    # Ferramentas do agente de exemplo (remova se não precisar)
    "langchain-tavily>=0.1",
    "langchain-openai>=1.2.2",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.9",
    "mypy>=1.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Wheel (.whl) é o formato padrão de distribuição de pacotes Python
[tool.hatch.build.targets.wheel]
packages = ["ai_platform"]
```

---

## Pontos de aprendizado

### O que é Wheel?

**Wheel** (`.whl`) é o formato padrão de distribuição de pacotes Python — pense nele como o equivalente ao **`.jar`** do Java.

| Java | Python |
|---|---|
| `.jar` (Java Archive) | `.whl` (Wheel) |
| Contém `.class` compilados | Contém `.py` + metadados |
| Distribuído via Maven/Gradle | Distribuído via PyPI/pip |
| `pom.xml` define o que entra | `pyproject.toml` define o que entra |

### Sobre as dependências

- **`mlflow[langchain]`**: instala MLflow com o extra `langchain`, que habilita `mlflow.langchain.autolog()` — a mágica que captura traces automaticamente.
- **`langchain-tavily`**: ferramenta de busca para agentes (opcional).
- **`langchain-openai`**: integração LangChain com API OpenAI-compatible (usado com LiteLLM Proxy).
- **`hatchling`**: build backend moderno para Python.

---

## Próxima etapa

➡️ [Etapa 2 — Configuração com Variáveis de Ambiente](ETAPA_02.md)
