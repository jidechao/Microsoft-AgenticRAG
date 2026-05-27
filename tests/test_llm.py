from types import SimpleNamespace

from agenticrag.llm import collect_stream_text


class Event:
    def __init__(self, content):
        self.choices = [
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
            ),
        ]


def test_collect_stream_text_yields_content_fragments_and_skips_none():
    events = [Event("你"), Event("好"), Event(None)]

    assert list(collect_stream_text(events)) == ["你", "好"]
