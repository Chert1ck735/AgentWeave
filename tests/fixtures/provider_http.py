"""Mock OpenAI Responses HTTP transport: no API key, no network."""
import json
import httpx


def sse(events: list[dict]) -> bytes:
    """Encode Responses streaming events as server-sent events."""
    return b''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'.encode() for event in events)


def text_response(text: str, *, usage: dict | None = {'input_tokens': 7, 'output_tokens': 3}) -> list[dict]:
    message = {'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'status': 'completed',
               'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}
    middle = len(text) // 2
    deltas = [text[:middle], text[middle:]]
    return ([{'type': 'response.output_text.delta', 'item_id': 'msg_1', 'output_index': 0,
              'content_index': 0, 'delta': part, 'sequence_number': i, 'logprobs': []}
             for i, part in enumerate(deltas) if part] +
            [_completed([message], usage)])


def tool_response(name: str, arguments: str, *, call_id: str = 'call_1',
                  usage: dict | None = {'input_tokens': 11, 'output_tokens': 5}) -> list[dict]:
    call = {'id': 'fc_1', 'type': 'function_call', 'call_id': call_id, 'name': name,
            'arguments': arguments, 'status': 'completed'}
    return [{'type': 'response.function_call_arguments.delta', 'item_id': 'fc_1', 'output_index': 0,
             'delta': arguments, 'sequence_number': 0},
            _completed([call], usage)]


def _completed(output: list[dict], usage: dict | None) -> dict:
    response = {'id': 'resp_1', 'object': 'response', 'created_at': 0, 'model': 'test-model',
                'status': 'completed', 'output': output, 'parallel_tool_calls': True,
                'tool_choice': 'auto', 'tools': []}
    if usage is not None:
        response['usage'] = usage | {'total_tokens': usage['input_tokens'] + usage['output_tokens'],
                                     'input_tokens_details': {'cached_tokens': 0},
                                     'output_tokens_details': {'reasoning_tokens': 0}}
    return {'type': 'response.completed', 'response': response, 'sequence_number': 99}


class OpenAIServer:
    """Scripted replies; records every request body it received."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=sse(reply))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def error(status: int, message: str = 'error') -> httpx.Response:
    return httpx.Response(status, json={'error': {'message': message, 'type': 'error', 'code': None}})
