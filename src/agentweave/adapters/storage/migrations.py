"""Versioned SQL resources; callers hold the backend migration transaction."""
from importlib.resources import files


def initial_statements() -> list[str]:
    return statements(1)


def statements(version: int) -> list[str]:
    names = {1:'001_initial.sql', 2:'002_directory.sql', 3:'003_runtime.sql', 4:'004_delivery_and_errors.sql', 5:'005_tool_executions.sql', 6:'006_model_attempts.sql', 7:'007_attempt_outcomes.sql', 8:'008_tool_catalogs.sql', 9:'009_conversations.sql', 10:'010_operations.sql'}
    source = files('agentweave.adapters.storage').joinpath('sql/sqlite/' + names[version]).read_text(encoding='utf-8')
    return [statement.strip() for statement in source.split(';') if statement.strip()]
