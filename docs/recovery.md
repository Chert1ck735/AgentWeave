# Восстановление и эксплуатация

## Гарантии

- Вход подтверждается брокеру после записи в журнал. Повтор доставки дедуплицируется
  по event_id и не создаёт второй задачи.
- Шаги модели и вызовы инструментов журналируются. После перезапуска цикл повторяет
  журнал, завершённые шаги заново не выполняются.
- Исходящие события и ответы уходят через outbox. Получатель дедуплицирует повторы.
- Попытка, прерванная после возможного эффекта и не помеченная `retry='safe'`,
  становится `outcome_unknown` и ждёт решения оператора.
- Dev-брокер не долговечен: событие, которое ни один агент не принял в журнал до
  перезапуска брокера, теряется. Принятые задачи хранятся в журнале (`doctor` сообщает
  о долговечности профиля).

## Действия оператора

```bash
agentweave task status <task-id>
agentweave task resolve <task-id> --step <tool-call-id> --expected-version <n> --result '12.5'
agentweave task retry <task-id> --step handler --expected-version <n> --allow-unknown
agentweave task fail <task-id> --expected-version <n> --reason "refunded by hand"
agentweave task cancel <task-id> --expected-version <n> --reason "not needed"
agentweave task result <task-id>
```

Каждое действие требует версию из `task status` и записывается в журнал аудита.
Подробности — docs/configuration.md.

## Состояние и обслуживание

```bash
agentweave state inspect <session-id>
agentweave state export backup.sqlite3
agentweave state cleanup
agentweave state cleanup --apply
```
