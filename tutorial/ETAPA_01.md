# Etapa 1 — Estrutura do Projeto e Configuração

## Visão Geral da Arquitetura

Antes de codar, vamos entender **o que** estamos construindo e **por quê**.

Este projeto é um **SDK/template** que resolve um problema simples: permitir que desenvolvedores criem agentes de IA e automaticamente tenham **rastreabilidade completa** via MLflow — sem precisar entender MLflow.

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
mkdir -p meu-agente/ai_platform
mkdir -p meu-agente/examples
cd meu-agente
```

Estrutura alvo:

```
meu-agente/
├── ai_platform/          # SDK da plataforma (o que vamos construir)
│   ├── __init__.py
│   ├── config.py
│   └── tracking.py
│
├── examples/
│   └── agent.py          # agente de exemplo
│
├── .env.example
├── .gitignore
├── docker-compose.yaml
├── Justfile
└── pyproject.toml
```

---

## Passo 1.2 — Criar o `pyproject.toml`

```toml
[project]
name = "hiae-ai-agent-template"
version = "0.1.0"
description = "Template de referência para agentes de IA generativa"
requires-python = ">=3.12"
dependencies = [
    # Plataforma AI (não remover)
    "mlflow[langchain]>=3.0,<4.0",
    "python-dotenv>=1.0",
    # Framework de agentes
    "langchain>=0.3",
    "langchain-core>=0.3",
    "langgraph>=0.2",
    "langchain-community>=0.3",
    "langchain-openai>=1.2.2",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.9",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

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

Quando o `pyproject.toml` diz `packages = ["ai_platform"]`, é como se no `pom.xml` do Java você dissesse: "no meu `.jar`, inclua **apenas** o pacote `com.hiae.ai_platform`". A pasta `examples/` e os testes **não** entram no pacote distribuído.

### Sobre as dependências

- **`mlflow[langchain]`**: instala MLflow com o extra `langchain`, que habilita `mlflow.langchain.autolog()` — a mágica que captura traces automaticamente.
- **`hatchling`**: build backend moderno para Python.

---

## Próxima etapa

➡️ [Etapa 2 — Configuração com Variáveis de Ambiente](ETAPA_02.md)
