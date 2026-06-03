# Etapa 9 — Model Registry: Empacotando e Versionando o Agente no MLflow

## O que vamos aprender

- Empacotar o agente como um modelo MLflow PyFunc (`PythonModel`)
- Registrar o modelo no **MLflow Model Registry** com versionamento automático
- Recuperar (carregar) o modelo registrado
- Testar o modelo carregado chamando `.predict()`
- Entender por que o MLflow **não sobrescreve** — ele versiona

---

## Conceito

Até agora, o agente era executado diretamente como um script Python. O **Model Registry** transforma o agente em um **artefato versionado e portável**:

```
Antes (Etapas 1–8)          Depois (Etapa 9)
┌─────────────────┐          ┌──────────────────────────────────┐
│ python           │          │ MLflow Model Registry            │
│   agents/        │  ──────▶ │                                  │
│   agent.py       │          │  t-zero.d-zero.a-zero            │
│                  │          │  ├── Version 1  (agents/agent.py)│
│  (script local)  │          │  ├── Version 2  (agente melhorado│
└─────────────────┘          │  └── Version 3  ← latest         │
                              └──────────────────────────────────┘
```

### Por que registrar no Model Registry?

| Sem Registry | Com Registry |
|---|---|
| Agente existe só como arquivo `.py` | Agente é um artefato versionado e rastreável |
| Para atualizar, substitui o arquivo | Cada atualização cria uma nova versão (sem perder o histórico) |
| Não há rollback | Pode voltar para qualquer versão anterior |
| Difícil auditar qual versão estava em produção | Cada versão é imutável e rastreável |

---

## Arquitetura dos novos arquivos

```
model/
├── wrapper_model.py    ← AgentPyfuncWrapper (empacotador)
├── register_model.py   ← empacota e envia ao MLflow
├── load_model.py       ← baixa do MLflow Registry
└── ztest_model.py      ← fluxo completo: enviar + recuperar + testar
```

---

## Passo 9.1 — Criar o Wrapper PyFunc

O MLflow exige que modelos customizados implementem `mlflow.pyfunc.PythonModel`. Crie `model/wrapper_model.py`:

```python
"""
Wrapper PyFunc para empacotar o agente como modelo MLflow.
"""

import importlib.util

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel


class AgentPyfuncWrapper(PythonModel):

    def load_context(self, context):
        """
        Carrega o módulo do agente a partir do artefato 'agent_file'.
        Chamado UMA VEZ quando o modelo é inicializado.
        """
        agent_file = context.artifacts["agent_file"]

        spec = importlib.util.spec_from_file_location("user_agent", agent_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.agent_module = module

    def predict(self, context, model_input):
        """
        Executa o agente e retorna a resposta como DataFrame.
        Chamado a cada inferência.
        """
        if isinstance(model_input, pd.DataFrame):
            message = model_input["message"].iloc[0]
        elif isinstance(model_input, dict):
            message = model_input["message"]
        else:
            raise ValueError("model_input deve ter campo 'message'")

        result = self.agent_module.invoke_agent(message)

        if isinstance(result, dict):
            answer = result.get("content", str(result))
        else:
            answer = str(result)

        return pd.DataFrame([{"answer": answer}])
```

### O que cada método faz

| Método | Quando é chamado | O que faz |
|---|---|---|
| `load_context()` | Uma vez, ao carregar o modelo | Importa `agents/agent.py` dinamicamente via `importlib` |
| `predict()` | A cada chamada de inferência | Extrai a mensagem, chama `invoke_agent()`, retorna DataFrame |

### Por que `importlib`?

O MLflow **copia** o arquivo do agente para dentro do artifact store quando o modelo é registrado. Na hora de carregar, o arquivo está em um diretório temporário gerenciado pelo MLflow — não no `sys.path` normal. O `importlib` permite importar qualquer arquivo `.py` pelo caminho absoluto, sem precisar que esteja no `sys.path`.

```
Registro                          Carregamento
agents/agent.py  ──copia──▶  mlflow_data/artifacts/.../agent.py
                                        │
                              context.artifacts["agent_file"]
                                        │
                              importlib.util.spec_from_file_location(...)
```

---

## Passo 9.2 — Criar o `register_model.py`

Crie `model/register_model.py`. Este arquivo empacota o agente e o envia ao MLflow:

```python
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from model.wrapper_model import AgentPyfuncWrapper

load_dotenv()


def _artifact_path() -> str:
    """Lê ARTIFACT_PATH do .env — nome do diretório do artefato na run."""
    path = os.getenv("ARTIFACT_PATH")
    if not path:
        raise EnvironmentError("ARTIFACT_PATH não definido no .env")
    return path


def register_model(agent_file: str) -> mlflow.models.model.ModelInfo:
    """Empacota o agente e registra no MLflow Model Registry."""

    if not agent_file:
        raise ValueError("agent_file é obrigatório. Exemplo: 'agents/agent.py'")

    agent_file = str(Path(agent_file).resolve())

    if not Path(agent_file).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {agent_file}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

    input_example = pd.DataFrame([{"message": "Qual a capital do Brasil?"}])
    output_example = pd.DataFrame([{"answer": "Brasília é a capital do Brasil."}])
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path=_artifact_path(),   # ← lido do .env (ARTIFACT_PATH)
            python_model=AgentPyfuncWrapper(),
            artifacts={"agent_file": agent_file},
            registered_model_name=f"{os.getenv('TEAM_NAME')}.{os.getenv('DOMAIN')}.{os.getenv('AGENT_NAME')}",
            input_example=input_example,
            signature=signature,
        )

    return model_info
```

### O que `artifact_path` significa

O `artifact_path` é o **nome do subdiretório** dentro da run onde o modelo é armazenado:

```
runs/<run_id>/
└── agente-do-angelo/        ← artifact_path (valor de ARTIFACT_PATH no .env)
    ├── MLmodel
    ├── python_model.pkl
    └── artifacts/
        └── agent_file/
            └── agent.py     ← cópia do agents/agent.py
```

O `model_uri` retornado será: `runs:/<run_id>/agente-do-angelo`

---

## Passo 9.3 — Criar o `load_model.py`

Crie `model/load_model.py`. Este arquivo baixa o modelo do Registry:

```python
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mlflow
from dotenv import load_dotenv

load_dotenv()


def load_model(model_uri: str | None = None) -> mlflow.pyfunc.PyFuncModel:
    """
    Baixa e carrega um modelo PyFunc do MLflow Model Registry.

    Args:
        model_uri: URI do modelo. Se None, usa a última versão registrada.
                   Formatos:
                   - "models:/<nome>/latest"   → última versão
                   - "models:/<nome>/1"        → versão específica
                   - "runs:/<run_id>/agent"    → direto de uma run
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))

    if model_uri is None:
        agent_name = os.getenv("AGENT_NAME")
        team = os.getenv("TEAM_NAME", "default")
        domain = os.getenv("DOMAIN", "geral")
        model_uri = f"models:/{team}.{domain}.{agent_name}/latest"

    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model
```

---

## Passo 9.4 — Criar o `ztest_model.py` (fluxo completo)

Crie `model/ztest_model.py`. Este arquivo orquestra o fluxo completo:

```python
"""
Fluxo completo: enviar → recuperar → testar.

Uso:
    uv run python model/ztest_model.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mlflow
import pandas as pd
from dotenv import load_dotenv
from model.load_model import load_model
from model.register_model import register_model

load_dotenv()


def send_model(agent_file: str) -> mlflow.models.model.ModelInfo:
    """Empacota e envia o agente ao MLflow. Deve ser chamado ANTES de test_model()."""
    model_info = register_model(agent_file=agent_file)
    return model_info


def test_model(
    model_uri: str | None = None,
    message: str = "Qual a capital do Brasil?",
) -> pd.DataFrame:
    """Carrega o modelo do MLflow e executa .predict()."""
    loaded_model = load_model(model_uri=model_uri)
    input_df = pd.DataFrame([{"message": message}])
    result = loaded_model.predict(input_df)
    print(f"\n  Resposta: {result['answer'].iloc[0]}\n")
    return result


if __name__ == "__main__":
    # 1. Empacota o agente e registra no MLflow Model Registry
    model_info = send_model(agent_file="agents/agent.py")

    # 2. Baixa o modelo registrado e testa chamando .predict()
    test_model(
        model_uri=model_info.model_uri,
        message="Qual a capital do Brasil?",
    )
```

---

## Passo 9.5 — Configurar o `.env`

Adicione a variável `ARTIFACT_PATH` no seu `.env`:

```env
ARTIFACT_PATH=agente-do-angelo
```

O valor define o nome do subdiretório do artefato dentro da run MLflow.

---

## Passo 9.6 — Executar o fluxo completo

```bash
# 1. MLflow rodando
docker compose -f docker/docker-compose.yml up -d

# 2. Certifique-se que há pelo menos uma run dev (agents/agent_mock.py)
uv run python agents/agent_mock.py "Qual a capital do Brasil?"

# 3. Executar o fluxo completo: enviar + recuperar + testar
uv run python model/ztest_model.py
```

Saída esperada:

```
═══════════════════════════════════════════════════════
  📦 ETAPA 1 — ENVIANDO MODELO PARA O MLFLOW
═══════════════════════════════════════════════════════

  ✅ MODELO REGISTRADO COM SUCESSO
  Agent name         : a-zero
  Registered model   : t-zero.d-zero.a-zero
  Model URI          : runs:/<run_id>/agente-do-angelo

═══════════════════════════════════════════════════════
  🔽 ETAPA 2 — RECUPERANDO MODELO DO MLFLOW
═══════════════════════════════════════════════════════
  ✅ Modelo carregado com sucesso.

═══════════════════════════════════════════════════════
  🧪 ETAPA 3 — EXECUTANDO AGENTE SALVO NO MLFLOW
═══════════════════════════════════════════════════════
  Mensagem   : Qual a capital do Brasil?

  Resposta: Brasília é a capital do Brasil.
```

---

## O que você verá no MLflow UI (http://localhost:5050)

```
Models (menu lateral)
└── t-zero.d-zero.a-zero
    ├── Version 1  ← primeiro registro
    ├── Version 2  ← segundo registro (nova versão do agente)
    └── Version 3  ← latest
```

Cada versão é **imutável** — o MLflow nunca sobrescreve. Isso garante:
- **Rollback**: voltar para qualquer versão anterior
- **Auditoria**: saber exatamente qual versão estava em produção
- **Rastreabilidade**: cada versão aponta para a run que a gerou

---

## Por que o MLflow não sobrescreve?

O MLflow segue o princípio de **imutabilidade de artefatos de ML**:

| Comportamento | Motivo |
|---|---|
| Cria nova versão a cada registro | Preserva histórico completo |
| Versões são imutáveis | Garante reprodutibilidade |
| `latest` aponta para a mais recente | Facilita o uso sem precisar saber o número |
| Aliases (`@production`, `@staging`) | Permite promoção controlada entre ambientes |

---

## Referência rápida — URIs de modelo

| URI | O que carrega |
|---|---|
| `models:/t-zero.d-zero.a-zero/latest` | Última versão registrada |
| `models:/t-zero.d-zero.a-zero/1` | Versão específica (v1) |
| `models:/t-zero.d-zero.a-zero@production` | Alias (ex: produção) |
| `runs:/<run_id>/agente-do-angelo` | Direto de uma run específica |

---

## Próximos passos

```
✅ Etapa 9 — Model Registry (empacotamento + versionamento)
    ↓
⏭️ Próximos passos:
    ├── Aliases e promoção entre ambientes (dev → staging → prod)
    ├── Tools + LangGraph (agentes multi-step)
    └── CI/CD com registro automático a cada merge
```
