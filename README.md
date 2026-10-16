<h1 align="center">🌀 🤖 AgentWeave</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.12–3.14">
  <img src="https://img.shields.io/badge/platform-Linux%20x86__64-555?style=flat-square&logo=linux&logoColor=white" alt="Linux x86_64">
  <img src="https://img.shields.io/badge/license-MIT-2ea44f?style=flat-square" alt="Лицензия MIT">
</p>

SDK и CLI для независимых событийных агентов на Python. Каждый агент — отдельный процесс:
он получает запросы и события через брокер, записывает каждую задачу в журнал до того,
как что-то сделать, вызывает инструменты, модели и других агентов и после сбоя продолжает
с того места, где остановился, не повторяя подтверждённых шагов.

<br>

### 🌀 Что умеет AgentWeave

- **Агенты как процессы.** Запускаются одной командой `agentweave dev run`, живут независимо
  и находят друг друга и инструменты через брокер. Локальный брокер уже внутри пакета:
  Docker, Rust и JVM не нужны.

- **Журнал прежде эффекта.** Вход, шаги модели, вызовы инструментов и исходящие сообщения
  проходят через журнал SQLite. Перезапуск, повторная доставка и вторая реплика не создают
  двойной работы.

<br>

## Архитектура

```
           Client (Python)  ·  CLI agentweave
                 запросы · события · чат
                            │
                            ▼
    ┌───────────────────────────────────────────────┐
    │                     Брокер                    │
    │    встроенный, без Docker · каталог агентов   │
    └───────────────────────────────────────────────┘
               │                         │
               ▼                         ▼
    ┌─────────────────────┐   ┌─────────────────────┐
    │       Агент 1       │◀─▶│       Агент 2       │
    │     ИИ · OpenAI     │   │     ИИ · OpenAI     │
    └─────────────────────┘   └─────────────────────┘
               │                         │
               └────────────┬────────────┘
                            │  вызовы инструментов ⇄ результаты
                            ▼
    ┌───────────────────────────────────────────────┐
    │                  Tool-сервис                  │
    │     Python · HTTP · MCP, общий для агентов    │
    └───────────────────────────────────────────────┘

           каждый шаг сначала пишется в журнал
    ┌───────────────────────────────────────────────┐
    │     Журнал SQLite (WAL, synchronous=FULL)     │
    │  приём · lease · шаги модели · outbox · аудит │
    └───────────────────────────────────────────────┘
```

Клиент или CLI отправляет **запрос** конкретному агенту или публикует **событие** в тему,
и каждый подписанный агент получает его независимо. Стрелка `◀─▶` между агентами —
consult и handoff: агенты спрашивают друг друга и передают задачи.
Агент сначала **принимает** вход в журнал (с дедупликацией по event_id), затем берёт
задачу под **lease** и выполняет обработчик. Для ИИ-агента это управляемый цикл: каждый шаг
модели и каждый вызов инструмента журналируется, а уже выполненные шаги после перезапуска
не повторяются. Ответы и исходящие события уходят через **outbox** и доставляются, даже если
процесс упал сразу после фиксации результата.

Ключевые решения:
- **Один владелец задачи.** Lease и fencing не дают двум репликам зафиксировать одну задачу,
  а опоздавший владелец не может записать результат.
- **Честная неопределённость.** Если попытка оборвалась после возможного внешнего эффекта,
  а обработчик не помечен `retry='safe'`, задача становится `outcome_unknown` и ждёт решения
  оператора. Повторять её вслепую SDK не станет.
- **Инструменты как сервисы.** Tool-сервис публикует инструменты в живой каталог, и агент
  с `Tools.discover()` видит их без перезапуска. Вызов идёт через брокер и журнал.
- **Агенты зовут агентов.** `ctx.consult` спрашивает соседа с общим бюджетом и защитой от
  циклов, а `ctx.handoff` передаёт задачу или весь диалог, сохраняя ровно одного владельца.
- **Модели по выбору.** OpenAI, любые OpenAI-совместимые endpoints и собственные адаптеры.
  Пул запросов к провайдеру общий для всех процессов профиля, бюджеты токенов и стоимости
  общие для корневой задачи и всех её consult.

<br>

## Требования

- Linux x86_64 с glibc 2.39+ (например Ubuntu 24.04).
- Python 3.12, 3.13 или 3.14. Hash-lock и wheelhouse собраны для 3.12; 3.13 и 3.14
  проверяются тестами в CI.
- [uv](https://docs.astral.sh/uv/) — менеджер пакетов Python.
- `curl` и `binutils` (`readelf`) — только для сборки: скачать и проверить закреплённый релиз брокера.
- Ключ API провайдера модели нужен только ИИ-агентам. Примеры работают без ключа и без сети.

<br>

### 1. Установите uv

Если `uv` ещё не установлен:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

После установки перезапустите терминал.

<br>

### 2. Соберите пакеты

Пакеты не публикуются в индекс. Их собирают из исходников в локальный wheelhouse:
оба wheel AgentWeave и wheel всех зависимостей, сверенные с hash-lock. Сеть нужна только
на этом шаге.

<details>
<summary><strong>Вариант A: установка из wheelhouse (рекомендуется)</strong></summary>

```bash
git clone <адрес-репозитория> AgentWeave && cd AgentWeave   # адрес появится после публикации
url=$(python3 -c "import json; print(json.load(open('packages/agentweave-dev/src/agentweave_dev/manifest.json'))['release_url'])")
curl --fail --location --output /tmp/tansu.tar.gz "$url"
uv run --no-project --python 3.12 python scripts/build_dev_broker.py --archive /tmp/tansu.tar.gz
uv run --no-project --python 3.12 python scripts/build_wheelhouse.py
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install --offline --no-index --find-links dist/wheelhouse --find-links dist/wheelhouse/deps -c requirements/py312.lock "agentweave[dev]"
```

`build_dev_broker.py` скачанный архив не меняет. Он сверяет SHA256 архива и бинарника,
архитектуру ELF, требование к glibc и лицензии с закреплённым manifest и только потом кладёт
официальный релиз брокера в пакет `agentweave-dev`. В `dist/wheelhouse/SHA256SUMS` лежат
хеши всех собранных wheel.

</details>

<details>
<summary><strong>Вариант B: окружение для разработки</strong></summary>

Так же, как CI: зависимости из hash-lock, брокер упакован, код запускается из исходников.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --require-hashes -r requirements/py312.lock
url=$(python3 -c "import json; print(json.load(open('packages/agentweave-dev/src/agentweave_dev/manifest.json'))['release_url'])")
curl --fail --location --output /tmp/tansu.tar.gz "$url"
.venv/bin/python scripts/build_dev_broker.py --archive /tmp/tansu.tar.gz
export PYTHONPATH=$PWD/src:$PWD/packages/agentweave-dev/src:$PWD
alias agentweave='.venv/bin/python -m agentweave'
```

</details>

<br>

## Быстрый старт

Команды выполняются из корня репозитория. Первый `dev run` сам готовит профиль, журнал и
брокер: на этой машине до готовности агента проходит 1,8–2,0 с.

<br>

### 1. Запустите агента

Агент-счётчик из `examples/counter.py` хранит состояние сессии и считает принятые числа:

```bash
agentweave dev run examples.counter:counter --detach
agentweave agent status counter
```

<br>

### 2. Отправьте запрос и событие

```bash
agentweave agent request counter '[1, 2]' --session-id 00000000-0000-4000-8000-000000000001
agentweave event publish numbers.received '{"numbers": [3]}' --session-id 00000000-0000-4000-8000-000000000001 --wait counter
```

Запрос вернёт `output: 2`, событие в той же сессии — `output: 3`. Состояние пережило
обе задачи, потому что агент `stateful`.

<br>

### 3. Поговорите с агентом

```bash
agentweave agent chat counter
```

Каждая строка ввода — одно JSON-значение, например `[4]`. Ответ приходит строкой JSON
с `active_agent`. Строки, которые не являются JSON, отклоняются с `invalid_json`.

<br>

### 4. Остановите всё

```bash
agentweave agent stop counter
agentweave infra down
```

`infra down` останавливает брокер. Журнал `.agentweave/dev/state.sqlite3` сохраняется.

<br>

## Свой агент на Python

```python
from agentweave import Agent, Model, tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

assistant = Agent('assistant', model=Model.openai('<model-id>'), tools=[add])
```

```bash
export OPENAI_API_KEY=<ваш-ключ>
agentweave dev run app:assistant --chat
```

Агента без кода можно создать из CLI: определение записывается в `agentweave.toml`,
а ключ берётся из переменной окружения и в файлы не попадает.

```bash
agentweave agent create analyst --non-interactive --provider openai --model <model-id>
agentweave agent validate analyst
```

Пример без ключа и без сети: `python -m examples.assistant --offline`.

<br>

## Инструменты агентов

| Вид | Как подключить | Пример |
|------|-------------|--------|
| Python-функция | `@tool` и `tools=[...]` в определении агента | `examples/calculator.py` |
| Отдельный tool-сервис | `agentweave dev run <модуль:инструмент> --namespace <ns> --detach`, агент с `Tools.discover()` | `examples/calculator.py` |
| HTTP | Декларация в `agentweave.toml`, секреты — ссылками на переменные окружения | `examples/http_tool.toml` |
| MCP | stdio-сервер как дочерний процесс tool-сервиса, список инструментов отслеживается | `examples/mcp_toolbox.toml` |

<br>

## Команды CLI

| Группа | Команды | Назначение |
|------|---------|-------------|
| `dev` | `run` | Подготовить профиль и запустить агента или tool-сервис (`--detach`, `--chat`, `--replica`) |
| `agent` | `start` `stop` `drain` `restart` `status` `logs` `list` | Жизненный цикл процессов и реплик |
| `agent` | `create` `validate` `request` `chat` | Определения, запросы, диалоги |
| `event` | `publish` `subscribe` | Публикация событий (`--wait`, `--key`, `--expires-at`) и чтение темы |
| `task` | `status` `result` `events` `resolve` `retry` `fail` `cancel` | Состояние задач и действия оператора |
| `state` | `inspect` `export` `cleanup` | Память сессий, резервная копия журнала, retention |
| `tool` | `list` `describe` `stop` | Каталог инструментов |
| `infra` | `up` `down` `status` | Брокер профиля |
| `doctor` | — | Проверка окружения и долговечности профиля |

С `--json` каждая команда печатает по одному JSON-объекту на строку. Коды выхода:
`0` — успех, `2` — ошибка ввода, `3` — инфраструктура недоступна, `4` — задача
завершилась неуспешно (`failed`, `cancelled`, `expired`) или её исход неизвестен, `1` — прочее. Подробно — [docs/configuration.md](docs/configuration.md).

<br>

## Восстановление после сбоев

```bash
agentweave task status <task-id>
agentweave task resolve <task-id> --step <tool-call-id> --expected-version <n> --result '12.5'
```

Любое действие оператора требует версию из `task status` и записывается в журнал аудита.
Гарантии, сценарии `outcome_unknown` и обслуживание журнала описаны в [docs/recovery.md](docs/recovery.md).

<br>

## Ограничения

- Только Linux x86_64 и профиль `dev`. Серверного профиля пока нет.
- Dev-брокер не долговечен: событие, которое ни один агент не принял в журнал до
  перезапуска брокера, теряется. Принятые задачи хранятся в журнале.
- Адаптеры моделей: OpenAI, OpenAI-совместимые endpoints и собственные. Нативных
  адаптеров Anthropic и Gemini пока нет.

<br>

## Документация

| Документ | О чём |
|------|-------------|
| [Быстрый старт](docs/quickstart.md) | Первый агент, запросы, события, CLI |
| [Конфигурация](docs/configuration.md) | `agentweave.toml`, `agent create/validate`, коды выхода, retention |
| [Инструменты и модели](docs/tools.md) | Python, HTTP и MCP, OpenAI, совместимые endpoints, свои адаптеры |
| [Взаимодействие агентов](docs/coordination.md) | consult, handoff, диалоги |
| [Восстановление](docs/recovery.md) | Гарантии, `outcome_unknown`, действия оператора |
| [Изменения](CHANGELOG.md) | Что реализовано |

<br>

## Лицензия

Код распространяется по лицензии [MIT](LICENSE). Встроенный брокер Tansu распространяется
по лицензии Apache-2.0 ([packages/agentweave-dev/licenses](packages/agentweave-dev/licenses)).

<br>

<p align="center">
  <strong>🌀 AgentWeave</strong> — независимые агенты, которые помнят, что уже сделали.<br>
  Начните с <a href="docs/quickstart.md">быстрого старта</a> · примеры в <a href="examples">examples/</a>
</p>
