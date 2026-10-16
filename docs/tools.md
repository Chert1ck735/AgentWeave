# Инструменты и модели

## Python-инструмент

```python
from agentweave import Agent, Model, tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

assistant = Agent('assistant', model=Model.openai('<model-id>'), tools=[add])
```

Каждый вызов журналируется до эффекта. Инструмент без `retry='safe'` после сбоя во время
вызова не повторяется: задача становится `outcome_unknown` (см. docs/recovery.md).

## Отдельный tool-сервис

```bash
agentweave dev run examples.calculator:add --namespace calculator --detach
agentweave tool list
agentweave tool describe calculator.add
agentweave tool stop calculator.add
```

Агент с `tools=Tools.discover()` видит опубликованные инструменты без перезапуска.

## HTTP и MCP

HTTP-инструменты и MCP-серверы описываются в `agentweave.toml`
(примеры: examples/http_tool.toml, examples/mcp_toolbox.toml) и проверяются так:

```bash
agentweave agent validate weather
agentweave dev run weather --detach
```

## Модели

- `Model.openai(model_id)` — Responses API.
- `Model.compatible(model_id, base_url=...)` — OpenAI-совместимые endpoints (Chat Completions).
- `Model.custom(adapter)` или `factory_entrypoint` — свой адаптер; контракт проверяет
  `agentweave.testing.check_model_adapter` (пример: examples/custom_adapter.py).

Пул запросов к провайдеру общий для всех процессов профиля (`provider_pool_limit`).
Бюджеты токенов и стоимости (`ModelBudget`) общие для корневой задачи и её consult.
