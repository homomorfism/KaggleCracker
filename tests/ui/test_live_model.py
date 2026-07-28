"""The live adapter's pure parts — no network anywhere in the test suite."""

from ui.live_model import to_json_schema


def test_to_json_schema_maps_types_bounds_enums_defaults():
    schema = to_json_schema(
        {
            "path": {"type": "str", "required": True},
            "checks": {"type": "list", "required": True, "item_enum": ["a", "b"]},
            "sample_rows": {"type": "int", "min": 100, "max": 200, "default": 150},
            "mode": {"type": "str", "enum": ["x", "y"]},
            "rate": {"type": "float", "min": 0.0},
            "flag": {"type": "bool", "default": False},
        }
    )
    assert schema["type"] == "object"
    assert schema["required"] == ["path", "checks"]
    assert schema["properties"]["path"] == {"type": "string"}
    assert schema["properties"]["checks"] == {"type": "array", "items": {"enum": ["a", "b"]}}
    assert schema["properties"]["sample_rows"] == {
        "type": "integer", "minimum": 100, "maximum": 200, "default": 150,
    }
    assert schema["properties"]["mode"] == {"type": "string", "enum": ["x", "y"]}
    assert schema["properties"]["rate"] == {"type": "number", "minimum": 0.0}
    assert schema["properties"]["flag"] == {"type": "boolean", "default": False}
