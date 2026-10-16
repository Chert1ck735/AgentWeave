# Взаимодействие агентов

## Consult

```python
@agent.on_request(retry='safe')
async def ask(ctx, request):
    answer = await ctx.consult('analyst', request.input)
    return answer.output
```

Peer получает только этот вход в новой сессии. Пока handler ждёт, его слот свободен.
Корневой бюджет общий, циклы обнаруживаются (`CycleDetected`). Агенту нужен `peers`,
например `Peers.allow(['analyst'])`.

## Handoff

`ctx.handoff('analyst', message=...)` передаёт задачу другому агенту. Владелец у задачи
всегда один. В диалоге (`scope='conversation'`) следующий ход тоже уходит новому агенту.
Передач в одной цепочке не больше 8.

## Диалоги

```bash
agentweave agent chat general
agentweave agent chat general --conversation <conversation-id>
```

Каждая непустая строка stdin — одно JSON-значение: `"привет"` в кавычках, `[1, 2]` или `{...}`.
Строка не-JSON получает `invalid_json` и не отправляется, пустые строки пропускаются.
Ответ — строка JSON с результатом и `active_agent`. Конец stdin завершает только чат,
отправленные задачи продолжаются.
