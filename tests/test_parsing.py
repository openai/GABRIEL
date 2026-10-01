import pytest
from gabriel.utils.parsing import safe_json


def test_safe_json_with_comments():
    raw = (
        '[\n'
        '  {\n'
        '    "name": "Example",\n'
        '    "links": [\n'
        '      "https://example.com", // comment\n'
        '      "s3://bucket/file"  // another comment\n'
        '    ]\n'
        '  }\n'
        ']'
    )
    parsed = safe_json(raw)
    assert isinstance(parsed, list)
    assert parsed[0]["links"][0] == "https://example.com"



@pytest.mark.parametrize("value", [
    "keep /* this text */ intact",
    "a//b",
    'an escaped "quote" followed by // text',
    "https://example.com/path//part",
    "backslash \\ and /* literal */",
])
def test_safe_json_preserves_comment_markers_inside_strings(value):
    import json

    raw = '{"value": ' + json.dumps(value) + ', /* actual block comment */ "ok": true // actual line comment\n}'
    assert safe_json(raw) == {"value": value, "ok": True}


def test_safe_json_does_not_join_tokens_across_comments():
    assert safe_json('{"value": 1 /* separator */ 2}') == {}


def test_safe_json_handles_multiline_comments_in_arrays():
    assert safe_json('[1, /* first line\nsecond line */ 2, // last item\n3]') == [1, 2, 3]
