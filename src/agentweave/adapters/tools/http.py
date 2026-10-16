"""JSON HTTP(S) tools from configuration: a static target, mapped parameters, bounded responses.

The binding sends exactly one request per invocation and never retries: the
shared ToolExecutor owns retries, timeouts and the journal. Secrets are read
from the environment at call time and never appear in errors or logs.
"""
import json
import logging
import os
import re
from collections.abc import Mapping
from ...config import SecretRef
from ...definitions.toolbox import Argument, HttpToolSpec, Literal
from ...errors import (AgentWeaveError, ToolArgumentsInvalid, ToolNotStarted, ToolRequestRejected,
                       ToolResponseInvalid)
from ...messages import JSONValue

_JSON_TYPES = ('application/json',)
_QUERY = re.compile(r'\?[^\s"\']*')


class _QueryRedaction(logging.Filter):
    """httpx logs every request URL; a query string may carry a secret, so it is cut."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = _QUERY.sub('?<redacted>', record.getMessage()), ()
        return True


def _redact_client_logs() -> None:
    logger = logging.getLogger('httpx')
    if not any(isinstance(existing, _QueryRedaction) for existing in logger.filters):
        logger.addFilter(_QueryRedaction())


class HttpEndpointError(AgentWeaveError):
    """The endpoint failed (HTTP 5xx); it may have acted before failing."""
    def __init__(self, status: int):
        super().__init__(f'the tool endpoint failed with HTTP {status}')
        self.status = status


_redact_client_logs()  # before any handler sees an httpx record
_RETRY_LATER = frozenset((408, 429))  # the endpoint did not process the request
_BREAKS = re.compile(r'[\r\n\x00]')


def _reject_constant(value):
    raise ValueError('non-finite JSON number')


def _finite(text: str) -> float:
    value = float(text)
    if value in (float('inf'), float('-inf')):
        raise ValueError('non-finite JSON number')
    return value


def _unique_keys(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON object key')
        result[key] = value
    return result


def _text(value) -> str:
    """A query or header value: JSON scalars only, in their JSON spelling."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False)
    raise ToolArgumentsInvalid('query and header values must be strings, numbers or booleans')


class HttpBinding:
    def __init__(self, spec: HttpToolSpec, *, environ: Mapping[str, str] | None = None, transport=None) -> None:
        self.spec = spec
        self._environ = os.environ if environ is None else environ
        self._transport = transport

    def __repr__(self) -> str:
        return f'HttpBinding({self.spec.method} {self.spec.name})'

    async def invoke(self, call, context) -> JSONValue:
        """Send the call's approved arguments; raise why no usable JSON answer came back."""
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import best_match
        problem = best_match(Draft202012Validator(self.spec.input_schema).iter_errors(call.arguments))
        if problem is not None:
            raise ToolArgumentsInvalid(problem.message[:1000])
        query = {name: _text(value) for name, value in self._values(self.spec.query, call.arguments).items()}
        headers = self._headers(call.arguments)
        headers['Accept'] = 'application/json'
        headers['Accept-Encoding'] = 'identity'  # the byte limit counts what is actually received
        if self.spec.idempotency_header:
            # Stable across attempts; it does not make the endpoint idempotent by itself.
            headers[self.spec.idempotency_header] = call.tool_call_id
        body = self._values(self.spec.body, call.arguments)
        try:
            import httpx
        except ImportError:
            raise ToolNotStarted("HTTP tools need the http extra: pip install 'agentweave[http]'") from None
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.spec.timeout), trust_env=False,
                                     follow_redirects=False, transport=self._transport) as client:
            request = client.build_request(self.spec.method, self.spec.url, params=query or None, headers=headers,
                                           json=body if self.spec.body else None)
            try:
                response = await client.send(request, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.LocalProtocolError, httpx.InvalidURL,
                    httpx.UnsupportedProtocol) as error:
                # Refused before any byte of the request reached the endpoint.
                raise ToolNotStarted(f'could not reach the tool endpoint ({type(error).__name__})') from None
            except httpx.TimeoutException as error:
                raise TimeoutError(f'the tool endpoint did not answer in time ({type(error).__name__})') from None
            except httpx.HTTPError as error:
                raise ConnectionError(f'the tool request failed ({type(error).__name__})') from None
            try:
                return await self._read(response, httpx)
            finally:
                await response.aclose()

    def _headers(self, arguments: dict) -> dict[str, str]:
        """Header values without line breaks: a model argument is invalid, a broken secret stops the call."""
        headers = {}
        for name, value in self._values(self.spec.headers, arguments).items():
            text = _text(value)
            if _BREAKS.search(text):
                if isinstance(self.spec.headers[name], SecretRef):
                    raise ToolNotStarted(f'the secret for header {name} contains a line break')
                raise ToolArgumentsInvalid(f'header {name} may not contain line breaks')
            headers[name] = text
        return headers

    def _values(self, mapping: dict, arguments: dict) -> dict:
        values = {}
        for name, source in mapping.items():
            if isinstance(source, Literal):
                values[name] = source.value
            elif isinstance(source, Argument):
                if source.name in arguments:
                    values[name] = arguments[source.name]
            elif isinstance(source, SecretRef):
                secret = self._environ.get(source.env)
                if not secret:
                    raise ToolNotStarted(f'secret environment variable {source.env} is not set')
                values[name] = secret
        return values

    async def _read(self, response, httpx) -> JSONValue:
        status = response.status_code
        if 300 <= status < 400:
            # A redirect may follow a completed effect (post/redirect/get); it is not followed.
            raise ToolResponseInvalid(f'the tool endpoint redirected with HTTP {status}; redirects are not followed')
        if status in _RETRY_LATER:
            raise ToolNotStarted(f'the tool endpoint asked to retry later (HTTP {status})')
        if 400 <= status < 500:
            raise ToolRequestRejected(status, f'the tool endpoint refused the request with HTTP {status}')
        if status >= 500 or status < 200:
            raise HttpEndpointError(status)
        if response.headers.get('content-encoding', 'identity').strip().lower() not in ('', 'identity'):
            raise ToolResponseInvalid('the response is encoded; only identity responses are accepted')
        limit = self.spec.max_response_bytes
        declared = response.headers.get('content-length', '')
        if declared.isdigit() and int(declared) > limit:
            raise ToolResponseInvalid('the response exceeds max_response_bytes')
        raw = bytearray()
        try:
            async for chunk in response.aiter_raw():
                raw += chunk
                if len(raw) > limit:
                    raise ToolResponseInvalid('the response exceeds max_response_bytes')
        except httpx.TimeoutException as error:
            raise TimeoutError(f'the tool endpoint did not finish its answer in time ({type(error).__name__})') from None
        except httpx.HTTPError as error:
            raise ConnectionError(f'the tool response failed ({type(error).__name__})') from None
        if not raw:
            return None
        media = response.headers.get('content-type', '').split(';')[0].strip().lower()
        if media not in _JSON_TYPES and not media.endswith('+json'):
            raise ToolResponseInvalid('the tool endpoint did not answer with JSON')
        try:
            return json.loads(raw.decode('utf-8'), parse_constant=_reject_constant, parse_float=_finite,
                              object_pairs_hook=_unique_keys)
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise ToolResponseInvalid('the tool endpoint answered with invalid or non-strict JSON') from None


__all__ = ['HttpBinding', 'HttpEndpointError']
