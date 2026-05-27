from agenticrag.switcher import parse_switcher_decision


def test_parse_switcher_decision_accepts_simple_json():
    assert parse_switcher_decision('{"route": "simple"}') == "simple"


def test_parse_switcher_decision_accepts_complex_json_with_text():
    assert parse_switcher_decision('Result:\n{"route": "complex", "reason": "multi-doc"}') == "complex"


def test_parse_switcher_decision_defaults_to_complex_for_unknown():
    assert parse_switcher_decision("not json") == "complex"
