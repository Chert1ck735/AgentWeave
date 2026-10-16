# Changelog

## Первая версия

Профиль `dev` на Linux x86_64.

- Агенты на Python: обработчики запросов и событий, stateless/stateful-память,
  политики подписок (очередь, порядок по ключу, overflow `wait`/`reject`/`latest`).
- Журнал SQLite (WAL, `synchronous=FULL`): приём с дедупликацией, lease и fencing,
  outbox с квитанциями, восстановление после сбоев.
- Управляемый цикл ИИ: журналируемые шаги модели, общий пул провайдера и бюджеты,
  адаптеры OpenAI, OpenAI-совместимых endpoints и собственных моделей.
- Инструменты: Python, HTTP (декларативно), MCP; отдельные tool-сервисы с каталогом.
- Взаимодействие агентов: consult с общим бюджетом, handoff с одним владельцем, диалоги.
- CLI: `dev run`, `agent create/validate/start/stop/request/chat`, `event publish/subscribe`,
  `task status/resolve/retry/fail/cancel`, `state inspect/export/cleanup`, стабильные коды выхода.
- Проверки отказов, реплик и нагрузки на настоящем брокере и журнале.
- Лицензия MIT (LICENSE). Встроенный брокер Tansu — Apache-2.0.
