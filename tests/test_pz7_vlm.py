from __future__ import annotations

from pz7_vlm_gemini import extract_message_text, parse_json_array


def test_extract_message_text_handles_null_content() -> None:
    assert extract_message_text({"content": None}) == "[]"


def test_extract_message_text_handles_openrouter_text_parts() -> None:
    message = {"content": [{"type": "text", "text": '[{"object":"пистолет"}]'}]}

    assert extract_message_text(message) == '[{"object":"пистолет"}]'


def test_parse_json_array_handles_non_string_content() -> None:
    assert parse_json_array(None) == []
