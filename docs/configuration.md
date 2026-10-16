# AgentWeave: конфигурация агентов и ввод через CLI

Документ описывает профиль dev. Серверного профиля пока нет.

## Где хранится определение

Определение агента находится либо в Python (`module:object`), либо в таблице
`[agents.NAME]` файла `agentweave.toml` в корне проекта. Это два способа найти
одно и то же `Agent`: обе формы проходят одну валидацию SDK и при одинаковых
настройках дают одинаковый fingerprint.

```toml
[agents.analyst]
instructions = "Answer with numbers."
memory = "stateful"            # stateless (по умолчанию) или stateful
concurrency = 2
max_inflight = 5
model = { provider = "openai", model_id = "gpt-test" }   # api_key_env по умолчанию OPENAI_API_KEY
tools = { mode = "discover", allow = ["weather.*", "calc"] }
peers = { mode = "allow", names = ["writer"] }

[agents.analyst.budget]
max_tokens = 1000
max_cost_usd = "0.10"          # деньги — десятичная строка, не float

[agents.analyst.budget.pricing]
revision = "r1"
input_per_million_usd = "0.15"
output_per_million_usd = "0.60"
```

- Секреты не записываются: `api_key_env` — имя переменной окружения с ключом.
- `tools` — либо выбор обнаруженных инструментов (`mode = "discover"`, необязательный
  `allow`), либо список HTTP-инструментов в схеме `HttpToolSpec`. MCP-серверы
  описываются отдельными `[toolboxes.NAME]`, Python-функции публикует tool-сервис.
- Собственный адаптер модели: `provider = "custom"` и `factory_entrypoint` — имя из
  `register_adapter`, имя entry point группы `agentweave.models` или `module:object`.
- `max_cost_usd` требует явного тарифа `pricing`; тариф по имени модели не угадывается.
- Модель с `capabilities = { tools = false }` не может обслуживать `tools` и `peers`:
  consult и handoff тоже вызывают инструменты.

Политики подписок задаются для Python-обработчиков, а не продюсером события:

```toml
[agents.counter.subscription_policies."numbers.received"]
queue_limit = 3
concurrency = 2
ordering = "unordered"     # key (по умолчанию) или unordered
overflow = "latest"        # wait, reject или latest
replaceable = true         # latest допустим только для заменяемых снимков
```

Ограничения профиля общие для проекта, например
`[profiles.dev.limits] provider_pool_limit = 2`.

## Приоритет настроек

Явный override запуска выше TOML, TOML выше Python-определения, оно выше значений
по умолчанию. Overrides запуска: `--memory`, `--concurrency`, `--max-inflight`.
Этот порядок одинаков для `agent validate`, `agent run/start`, `dev run`
и Worker в SDK.

## agent create

```bash
agentweave agent create analyst --non-interactive --provider openai --model gpt-test --memory stateful
```

- Без `--non-interactive` команда спрашивает только недостающие значения:
  instructions, provider, model id, base URL (compatible) или factory (custom).
  Затем предлагает необязательные: переменную ключа, memory, tools и peers
  (`*` — все обнаруженные). Пустой ответ или конец ввода оставляет значение
  по умолчанию. Вопросы пишутся в stderr, ответы читаются из stdin.
- С `--non-interactive` отсутствие provider, model id, base URL или factory —
  ошибка `missing_value`.
- `--definition FILE` (`-` — stdin) принимает JSON-объект тех же полей, что и TOML;
  флаги переопределяют его значения, а выбор tools/peers из флагов заменяет выбор
  из файла целиком. Неизвестные поля и `name`, отличный от NAME, отклоняются.
- Существующее определение заменяется только с `--replace`.
- Файл заменяется атомарно под блокировкой `.agentweave/config.lock`. Параллельные
  create не теряют соседние определения и настройки профилей. Комментарии в
  `agentweave.toml` при перезаписи не сохраняются.

После создания агент запускается по имени: `agentweave dev run analyst --chat`.

## agent validate

```bash
agentweave agent validate app:helper --memory stateless
```

Команда загружает и разрешает определение так же, как запуск, но без инфраструктуры
и без запроса к модели. Она проверяет:
- схему, политики и бюджеты;
- совместимость возможностей модели;
- наличие переменной ключа и установленного адаптера.

В ответе — итоговые настройки и fingerprint. Ошибки содержат путь к полю
и причину, но не значения.

## Ввод: request, chat, event

- `agent request NAME [JSON]` печатает принятие (`task_id`, `accepted`), затем
  терминальный результат. `--no-wait` ограничивается принятием, `--session-id` —
  UUID сессии для stateful-агента.
- `agent chat NAME [--conversation ID]` ведёт один `Conversation`: строка stdin —
  одно сообщение. EOF завершает только чат, отправленные задачи продолжаются.
- `event publish TOPIC [JSON]` публикует событие через `Client.publish`
  (Envelope и EventBus), а не в обход admission. Метаданные: `--key`,
  `--event-time`, `--expires-at`, `--source-version`, `--session-id`.
  `--wait AGENT` дожидается результата обработки этим агентом
  (`--timeout`, по умолчанию 60 с).
- `event subscribe TOPIC` печатает события JSON-строками в собственной группе
  потребителей, поэтому не забирает события у агентов. Событие подтверждается
  только после печати. По умолчанию показываются события после начала подписки;
  `--from-retained` добавляет то, что ещё хранит брокер, `--limit N` останавливает
  после N событий. Ctrl-C или SIGTERM завершает подписку с кодом 0.

JSON берётся ровно из одного источника: аргумент, `--file` или stdin (когда нет
ни аргумента, ни файла). Допускается только строгий JSON в UTF-8: без NaN
и бесконечностей. Вход, имя для `--wait` и `--timeout` (конечное положительное
число) проверяются до подключения. Событие больше лимита конверта (1 МиБ) —
ошибка ввода. Команды ввода не
запускают брокер сами: без работающей инфраструктуры — `infrastructure_unavailable`.

### Что гарантирует публикация

- ACK брокера не означает приём агентом. Dev-брокер не долговечен: событие, которое
  ни один подписчик не принял до перезапуска брокера, теряется. Принятые задачи
  хранятся в журнале SDK. `doctor` показывает долговечность выбранного профиля.
- Coalescing происходит до admission, в очереди подписки. При политике
  `overflow = "latest"` с `replaceable = true` событие с `--key` и `--source-version`
  считается снимком. Ожидающий снимок с тем же отправителем, топиком и ключом
  заменяется более новым, а снимок с не большей версией отбрасывается. Решает
  конфигурация обработчика, не продюсер. `--expires-at` отбрасывает просроченные события.
- В dev нет постоянных offset потребителей и compaction. Полного replay истории
  нет: брокер хранит события ограниченное время, и AgentWeave их не переигрывает.

## Операторские действия: task

Команды `task` работают только с журналом SQLite профиля, брокер им не нужен.

- `task status ID` показывает задачу, её версию и шаги с видом (`tool`, `python_handler`,
  `model`). Неизвестный шаг помечен `outcome_unknown`. Значения аргументов под ключами
  вроде `token`, `password`, `api_key` скрыты (`***`).
- `task result ID` и `task events ID` — итог и сохранённый progress. Неуспешный
  итог — код 4.
- Каждое изменение требует `--expected-version` из `task status`. Устаревшая версия —
  `version_conflict`, поэтому два оператора не создают двух продолжений. Каждое
  действие записывается в журнал аудита `operator_actions`.
- `task resolve ID --step TOOL_CALL_ID --result JSON` записывает подтверждённый
  результат инструмента. Результат проверяется по схеме, с которой вызов был
  отправлен; затем цикл продолжается, завершённые шаги не повторяются.
  `--step handler` завершает весь Python-обработчик. `--state` задаёт полное
  подтверждённое состояние сессии.
- `task retry ID --step ... --allow-unknown` повторяет только неизвестный шаг
  (весь непрозрачный обработчик — с последнего подтверждённого состояния).
  Без `--allow-unknown` повтор отклоняется.
- Истёкший срок не продлевается сам: продолжение требует `--deadline` в будущем.
- `task fail ID --reason ...` завершает неработающую задачу ошибкой `OperatorFailed`.
- `task cancel ID --reason ...` отменяет сразу, если шаг не выполняется. Иначе
  новые шаги запрещены, а выполняющийся сообщает свой настоящий итог. Неизвестный
  итог отменой не закрывается: для него resolve, retry или fail. Если прерванная
  попытка, которую нельзя повторять, могла подействовать, отмена завершает задачу
  как `outcome_unknown`. Вызов удалённого инструмента отменяется через вызывающую задачу.

## Состояние и retention: state

- `state inspect SESSION_ID` — состояние, число сообщений, диалог и задачи сессии.
- `state export PATH` — согласованная копия через backup API SQLite, включая
  страницы в WAL. Существующий файл не перезаписывается.
- `state cleanup` — сначала отчёт (dry-run), удаление только с `--apply`, в одной
  сериализованной транзакции.

Сроки задаются в `[profiles.dev.retention]`: `terminal_days = 7`,
`idle_session_days = 30`, `replay_window_days = 7`.
- Результаты и детали завершённых задач удаляются через `terminal_days`.
  Незавершённые задачи, состояние, ожидания и outbox по сроку не удаляются.
- История stateful-сессии удаляется через `idle_session_days` без активности,
  если в сессии нет открытой задачи и не занят её session gate.
- ID дедупликации хранятся не меньше `replay_window_days + 1` дня, дольше для
  незавершённых задач. Событие старше окна replay или не новее уже удалённого
  ID отклоняется как `ReplayExpired`, а не исполняется повторно. Увеличение окна
  сразу продлевает горизонт следующей очистки. ID с ещё не доставленной квитанцией
  не удаляются.
- Пока любая задача того же корня не завершена, её дочерние задачи (consult,
  вызовы удалённых инструментов) не очищаются: повтор журнала корня читает их итоги.
- Событие, созданное больше чем на 5 минут «в будущем», не принимается: иначе
  неверные часы отправителя сдвинули бы границу дедупликации.

## Коды выхода

| Код | Значение | Примеры `error_code` |
| --- | --- | --- |
| 0 | успех | — |
| 2 | конфигурация или ввод | `usage_error`, `invalid_input`, `invalid_json`, `missing_value`, `definition_invalid`, `definition_exists`, `capability_mismatch`, `model_key_missing`, `agent_not_found`, `profile_invalid`, `action_refused`, `version_conflict`, `task_not_found` |
| 3 | инфраструктура недоступна | `infrastructure_unavailable`, `agent_unavailable`, `broker_unavailable`, `worker_start_failed`, `ownership_mismatch`, `dev_extra_missing`, `store_unavailable`, `journal_unavailable` |
| 4 | задача завершилась неуспешно или исход неизвестен | статус `failed`, `cancelled`, `expired`, `outcome_unknown`; `task_unknown` |
| 1 | прочие сбои | `operation_failed`, `drain_incomplete`, `stop_timeout` |
| 130 | прервано | `interrupted` |

С `--json` каждый ответ — одна JSON-строка. Ошибка — объект с `error_code`,
`message` и, если есть, `details`.
