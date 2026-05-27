from types import SimpleNamespace

from agenticrag.llm import DeepSeekClient
from agenticrag.llm import collect_stream_text


class Event:
    def __init__(self, content):
        self.choices = [
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
            ),
        ]


class MissingDeltaEvent:
    choices = [SimpleNamespace()]


class MissingContentEvent:
    choices = [SimpleNamespace(delta=SimpleNamespace())]


class NoChoicesEvent:
    choices = []


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make_client(response, model="deepseek-chat"):
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.model = model
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(response),
        ),
    )
    return client


def response_with_content(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            ),
        ],
    )


def test_collect_stream_text_yields_content_fragments_and_skips_falsy_content():
    events = [
        NoChoicesEvent(),
        MissingDeltaEvent(),
        MissingContentEvent(),
        Event(None),
        Event(""),
        Event("你"),
        Event("好"),
    ]

    assert list(collect_stream_text(events)) == ["你", "好"]


def test_complete_returns_content():
    messages = [{"role": "user", "content": "hello"}]
    client = make_client(response_with_content("answer"))

    assert client.complete(messages) == "answer"


def test_complete_returns_empty_string_for_empty_content():
    messages = [{"role": "user", "content": "hello"}]
    client = make_client(response_with_content(""))

    assert client.complete(messages) == ""


def test_complete_returns_empty_string_for_no_choices():
    messages = [{"role": "user", "content": "hello"}]
    client = make_client(SimpleNamespace(choices=[]))

    assert client.complete(messages) == ""


def test_stream_calls_create_with_stream_true_and_yields_text_chunks():
    messages = [{"role": "user", "content": "hello"}]
    events = [Event("你"), Event(""), Event("好"), Event(None)]
    client = make_client(events, model="test-model")

    assert list(client.stream(messages)) == ["你", "好"]
    assert client.client.chat.completions.calls == [
        {
            "model": "test-model",
            "messages": messages,
            "stream": True,
        },
    ]


def test_tool_call_passes_tools_and_returns_raw_response():
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    response = SimpleNamespace(raw=True)
    client = make_client(response, model="test-model")

    assert client.tool_call(messages, tools) is response
    assert client.client.chat.completions.calls == [
        {
            "model": "test-model",
            "messages": messages,
            "tools": tools,
            "stream": False,
        },
    ]
