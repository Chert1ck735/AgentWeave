"""The shared model adapter contract, FakeModel and lazy adapter creation."""
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest


def request():
    from agentweave.messages import Message
    from agentweave.models import ModelRequest
    return ModelRequest(messages=[Message(role='user', content='hi')], tools=[], model_id='fake',
                        deadline=datetime.now(timezone.utc) + timedelta(seconds=60))


async def test_fake_model_streams_scripted_turns_and_records_requests():
    from agentweave.models import TextDelta, TurnComplete
    from agentweave.testing import FakeModel, text_turn, tool_turn
    model = FakeModel([tool_turn('add', {'a': 1, 'b': 2}), text_turn('3')])
    first = [event async for event in model.generate(request())]
    second = [event async for event in model.generate(request())]
    assert isinstance(first[-1], TurnComplete) and first[-1].turn.tool_calls[0].name == 'add'
    assert ''.join(e.text for e in second if isinstance(e, TextDelta)) == '3'
    assert second[-1].turn.messages[-1].content == '3'
    assert len(model.requests) == 2


async def test_fake_model_raises_scripted_errors_and_reports_exhaustion():
    from agentweave.errors import ModelError
    from agentweave.testing import FakeModel
    model = FakeModel([ModelError('rate_limit', 'slow down', retryable=True)])
    with pytest.raises(ModelError):
        [event async for event in model.generate(request())]
    with pytest.raises(AssertionError):
        [event async for event in model.generate(request())]


def test_create_adapter_resolves_secrets_by_reference():
    from agentweave.errors import ConfigError, DefinitionError
    from agentweave.models import Model
    from agentweave.providers import create_adapter
    from agentweave.testing import FakeModel
    fake = FakeModel([])
    assert create_adapter(Model.custom(fake), {}) is fake
    with pytest.raises(ConfigError):
        create_adapter(Model.openai('m'), {})  # OPENAI_API_KEY reference not provided
    adapter = create_adapter(Model.openai('m'), {'OPENAI_API_KEY': 'sk-test'})
    assert type(adapter).__name__ == 'OpenAIAdapter'
    with pytest.raises(DefinitionError, match='v0.2'):
        create_adapter(Model.anthropic('m'), {'ANTHROPIC_API_KEY': 'x'})


def test_core_import_does_not_import_provider_sdks():
    code = ('import sys, agentweave, agentweave.providers, agentweave.testing; '
            'print(any(m == "openai" or m.startswith("openai.") for m in sys.modules))')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env={'PYTHONPATH': 'src', 'PATH': ''}, cwd=str(Path(__file__).parents[2]))
    assert result.stdout.strip() == 'False', result.stderr


def test_budget_requires_explicit_versioned_pricing_for_cost_limits():
    from decimal import Decimal
    from agentweave import Agent, Model
    from agentweave.models import ModelBudget, ModelPricing
    with pytest.raises(ValueError):
        ModelBudget(max_cost_usd='0.01')
    budget = ModelBudget(max_tokens=100, max_cost_usd='0.01',
                         pricing=ModelPricing(revision='2025-01', input_per_million_usd='1.5',
                                              output_per_million_usd='6'))
    agent = Agent('assistant', model=Model.openai('m'), budget=budget)
    assert agent.budget.pricing.cost(1_000_000, 1_000_000) == Decimal('7.5')
    assert agent.budget.pricing.cost(None, 10) is None  # unknown usage is never priced as zero
