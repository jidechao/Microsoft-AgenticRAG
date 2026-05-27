from agenticrag.prompts import SWITCHER_PROMPT
from agenticrag.switcher import classify_query, parse_switcher_decision


def test_parse_switcher_decision_accepts_simple_json():
    assert parse_switcher_decision('{"route": "simple"}') == "simple"


def test_parse_switcher_decision_accepts_complex_json_with_text():
    assert parse_switcher_decision('Result:\n{"route": "complex", "reason": "multi-doc"}') == "complex"


def test_parse_switcher_decision_defaults_to_complex_for_unknown():
    assert parse_switcher_decision("not json") == "complex"


def test_parse_switcher_decision_uses_first_valid_json_candidate():
    text = '{"route": "simple"}\nNote: {"extra": true}'
    assert parse_switcher_decision(text) == "simple"


def test_classify_query_sends_switcher_prompt_and_user_query():
    class FakeLLMClient:
        def __init__(self):
            self.messages = None

        def complete(self, *, messages):
            self.messages = messages
            return '{"route": "simple"}'

    llm_client = FakeLLMClient()

    assert classify_query(llm_client, "What is AgenticRAG?") == "simple"
    assert llm_client.messages == [
        {"role": "system", "content": SWITCHER_PROMPT},
        {"role": "user", "content": "What is AgenticRAG?"},
    ]
