# Быстрый старт

## Первый агент

```python
from agentweave import Agent

counter = Agent('counter', memory='stateful')


@counter.on_request(retry='safe')
async def count(ctx, request):
    ctx.state['count'] = ctx.state.get('count', 0) + len(request.input)
    return ctx.state['count']
```

Запуск: первый `dev run` сам готовит профиль, журнал и брокер.

```bash
agentweave dev run examples.counter:counter --detach
agentweave agent list
agentweave agent status counter
agentweave infra status
```

## Запросы и события

```bash
agentweave agent request counter '[1, 2]' --session-id 00000000-0000-4000-8000-000000000001
agentweave event publish numbers.received '{"numbers": [3]}' --session-id 00000000-0000-4000-8000-000000000001
agentweave event subscribe numbers.received --limit 1
agentweave agent chat counter
```

`agent chat` читает stdin: каждая непустая строка — одно JSON-значение (например `[4]`),
ответ — строка JSON. Подробнее — docs/coordination.md.

Из Python то же делает `Client`:

```python
from agentweave import Client

async with Client.connect() as client:
    result = await client.agent('counter').ask([1], session_id)
    event_id = await client.publish('numbers.received', {'numbers': [2]}, session_id=session_id)
```

## Агент без кода

```bash
agentweave agent create analyst --non-interactive --provider openai --model <model-id>
agentweave agent validate analyst
agentweave dev run analyst --chat
```

Ключ модели берётся из переменной окружения (`OPENAI_API_KEY` по умолчанию) и в файлы
не записывается. Пример без ключа и без сети: `python -m examples.assistant --offline`.

## Остановка

```bash
agentweave agent stop counter
agentweave infra down
agentweave doctor
```

`infra down` останавливает брокер. Журнал `.agentweave/dev/state.sqlite3` сохраняется.
