# Etapa 3 — O Coração: Integração com MLflow via `@track_agent`

## O que vamos aprender

- Como funciona o decorator pattern em Python
- API do MLflow: experiments, runs, tags, artifacts, autolog
- Como o `mlflow.langchain.autolog()` captura traces automaticamente

---

## Passo 3.1 — Criar `ai_platform/__init__.py`

```python
from ai_platform.tracking import track_agent

__all__ = ["track_agent"]
```

---

## Passo 3.2 — Construir `ai_platform/tracking.py` (passo a passo)

Vamos construir este arquivo **incrementalmente** para entender cada parte.

### Passo 3.2.1 — Esqueleto do decorator

```python
import functools
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def track_agent(func: F) -> F:
    """Decorator que registra execuções no MLflow."""
    @functools.wraps(func)
    def wrapper(message: str, **kwargs: Any) -> Any:
        # Por enquanto, só chama a função original
        result = func(message, **kwargs)
        return result
    return wrapper  # type: ignore[return-value]
```

**O que está acontecendo:**

- `track_agent` recebe uma função (`func`) e retorna uma nova função (`wrapper`) que "envolve" a original.
- `@functools.wraps(func)` preserva o nome e docstring da função original.
- `TypeVar("F")` é para type hints — diz ao mypy que o tipo de retorno é o mesmo tipo da função decorada.

### Analogia com Java/Spring

O `@track_agent` no Python faz a mesma coisa que `@Around` do Spring AOP — intercepta a chamada, executa código antes/depois, e chama o método original no meio.

```java
// Java/Spring equivalente:
@Aspect
@Component
public class TrackAgentAspect {

    @Around("@annotation(TrackAgent)")
    public Object trackExecution(ProceedingJoinPoint joinPoint) throws Throwable {
        String message = (String) joinPoint.getArgs()[0];

        // ANTES
        MLflow.startRun("meu-agente-dev");
        MLflow.logText(message, "input/user_message.txt");

        // Executa o método original (equivalente a func(message))
        Object result = joinPoint.proceed();

        // DEPOIS
        MLflow.logOutput(result);
        MLflow.endRun();

        return result;
    }
}
```

### Exemplo visual do "antes/depois"

**SEM o decorator:**

```python
def invoke_agent(message: str):
    result = f"Resposta para: {message}"
    return result

invoke_agent("Olá")
# Saída: "Resposta para: Olá"
# MLflow: NADA registrado ❌
```

**COM o decorator:**

```python
@track_agent
def invoke_agent(message: str):
    result = f"Resposta para: {message}"
    return result

invoke_agent("Olá")
```

O que **realmente** acontece:

```
┌─────────────────────────────────────────────────────┐
│  ANTES (código injetado pelo @track_agent)          │
│                                                     │
│  1. load_dotenv()                                   │
│  2. cfg = load_config()                             │
│  3. mlflow.set_tracking_uri("http://localhost:5000")│
│  4. mlflow.set_experiment("/time/dominio/agente")   │
│  5. mlflow.langchain.autolog(log_traces=True)       │
│  6. mlflow.start_run("meu-agente-dev")              │
│  7. mlflow.set_tags({team, domain, ...})            │
│  8. mlflow.log_text("Olá", "input/...")             │
├─────────────────────────────────────────────────────┤
│  SEU CÓDIGO ORIGINAL (intocado)                     │
│                                                     │
│  result = f"Resposta para: Olá"                     │
│  return result                                      │
├─────────────────────────────────────────────────────┤
│  DEPOIS (código injetado pelo @track_agent)         │
│                                                     │
│  9. mlflow.log_text(result, "output/...")            │
│  10. print("Run ID: abc123...")                      │
│  11. mlflow.end_run()                                │
└─────────────────────────────────────────────────────┘
```

**O ponto-chave:** seu código **não muda em nada**. O decorator apenas adiciona o setup e cleanup do MLflow em volta.

---

### Passo 3.2.2 — Adicionar setup do MLflow

```python
import mlflow
from dotenv import load_dotenv
from ai_platform.config import load_config


def _setup_mlflow(cfg) -> None:
    """Configura o MLflow: URI, experimento e autolog."""
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(_experiment_name(cfg))
    mlflow.langchain.autolog(log_traces=True)  # ← A MÁGICA


def _experiment_name(cfg) -> str:
    """Gera o nome do experimento: /<time>/<domínio>/<agente>"""
    return f"/{cfg.team_name}/{cfg.domain}/{cfg.agent_name}"
```

**O que cada linha faz:**

| Linha | O que faz |
|---|---|
| `set_tracking_uri` | Diz ao MLflow onde está o servidor (nosso Docker local) |
| `set_experiment` | Cria/seleciona o experimento com nome hierárquico `/<time>/<domínio>/<agente>` |
| `langchain.autolog(log_traces=True)` | **Linha mais importante!** Faz o MLflow "espionar" todas as chamadas LangChain e registrar traces automaticamente |

---

### Passo 3.2.3 — Completar o decorator com run e logging

```python
def track_agent(func: F) -> F:
    @functools.wraps(func)
    def wrapper(message: str, **kwargs: Any) -> Any:
        load_dotenv()
        cfg = load_config()
        _setup_mlflow(cfg)
        experiment = _experiment_name(cfg)

        with mlflow.start_run(run_name=f"{cfg.agent_name}-{cfg.environment}") as run:
            mlflow.set_tags({
                "ai_platform.agent_name": cfg.agent_name,
                "ai_platform.team": cfg.team_name,
                "ai_platform.domain": cfg.domain,
                "ai_platform.environment": cfg.environment,
                "ai_platform.framework": "langchain",
                "ai_platform.trace_enabled": "true",
            })

            mlflow.log_text(message, "input/user_message.txt")
            result = func(message, **kwargs)
            _log_output(result)
            _print_run_summary(run.info.run_id, experiment, cfg.mlflow_tracking_uri)

            return result
    return wrapper


def _log_output(result: Any) -> None:
    if isinstance(result, dict):
        mlflow.log_dict(result, "output/final_response.json")
    else:
        mlflow.log_text(str(result), "output/final_response.txt")


def _print_run_summary(run_id: str, experiment: str, tracking_uri: str) -> None:
    print(f"\n{'─' * 52}")
    print(f"  MLflow Run ID  : {run_id}")
    print(f"  Experimento    : {experiment}")
    print(f"  UI             : {tracking_uri}")
    print(f"{'─' * 52}\n")
```

---

## Referência rápida — API MLflow usada

| API MLflow | O que faz | Quando usar |
|---|---|---|
| `mlflow.start_run()` | Abre uma "sessão" de registro | Uma vez por execução do agente |
| `mlflow.set_tags()` | Metadados key-value para busca | Governança: time, domínio, agente |
| `mlflow.log_text()` | Salva texto como artifact | Input do usuário, output textual |
| `mlflow.log_dict()` | Salva dict como JSON artifact | Output estruturado |
| `mlflow.langchain.autolog()` | Captura automática de traces | Uma vez no setup — captura tudo |

---

## Próxima etapa

➡️ [Etapa 4 — Infraestrutura: MLflow via Docker](ETAPA_04.md)
