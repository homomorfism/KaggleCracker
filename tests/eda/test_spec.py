"""validate_panels: every panel type has a valid shape and a named rejection."""

from tools.eda.spec import (
    MAX_HEATMAP_DIM,
    MAX_PANELS,
    MAX_SCATTER_POINTS,
    MAX_TABLE_ROWS,
    validate_panels,
)


def _panel(ptype, data, pid="p1"):
    return {"id": pid, "type": ptype, "title": "T", "commentary": "c", "data": data}


VALID = {
    "histogram": {"column": "age", "bins": [{"x0": 0, "x1": 10, "count": 4}]},
    "bar": {"categories": ["a", "b"], "series": [{"name": "n", "values": [1, 2]}]},
    "scatter": {"x_label": "x", "y_label": "y", "points": [[1, 2], [3, 4.5]]},
    "heatmap": {"x_labels": ["a"], "y_labels": ["b", "c"], "values": [[1], [None]]},
    "line": {"x": [1, 2], "series": [{"name": "n", "values": [3, 4]}]},
    "table": {"columns": ["c1"], "rows": [["v"]]},
    "stat": {"label": "rows", "value": 891},
    "markdown": {"text": "# notes"},
}


def test_every_type_has_a_passing_shape():
    panels = [_panel(t, d, pid="p-%s" % t) for t, d in VALID.items()]
    assert validate_panels(panels) is None


def test_rejections_name_the_offending_panel():
    cases = [
        (_panel("histogram", {"bins": []}), "bins"),
        (_panel("bar", {"categories": ["a"], "series": [{"name": "n", "values": [1, 2]}]}), "length"),
        (_panel("scatter", {"points": [[1, "y"]]}), "point"),
        (_panel("heatmap", {"x_labels": ["a"], "y_labels": ["b"], "values": [[1, 2]]}), "row"),
        (_panel("table", {"columns": ["a"], "rows": [["x", "y"]]}), "cell"),
        (_panel("stat", {"value": 3}), "label"),
        (_panel("markdown", {"text": "  "}), "text"),
        (_panel("pie", {"x": 1}), "type"),
    ]
    for panel, needle in cases:
        problem = validate_panels([panel])
        assert problem is not None and "p1" in problem or needle in problem
        assert needle in problem, (panel["type"], problem)


def test_caps_are_enforced():
    too_many = [
        _panel("stat", {"label": "l", "value": 1}, pid="p%d" % i)
        for i in range(MAX_PANELS + 1)
    ]
    assert "max %d" % MAX_PANELS in validate_panels(too_many)

    fat_scatter = _panel(
        "scatter", {"points": [[i, i] for i in range(MAX_SCATTER_POINTS + 1)]}
    )
    assert "downsample" in validate_panels([fat_scatter])

    dim = MAX_HEATMAP_DIM + 1
    fat_heat = _panel(
        "heatmap",
        {"x_labels": ["x"] * dim, "y_labels": ["y"], "values": [[0] * dim]},
    )
    assert validate_panels([fat_heat]) is not None

    fat_table = _panel(
        "table", {"columns": ["c"], "rows": [["v"]] * (MAX_TABLE_ROWS + 1)}
    )
    assert "aggregate" in validate_panels([fat_table])


def test_duplicate_and_missing_ids_rejected():
    a = _panel("stat", {"label": "l", "value": 1}, pid="same")
    b = _panel("stat", {"label": "l", "value": 2}, pid="same")
    assert "duplicate" in validate_panels([a, b])

    nameless = {"type": "stat", "title": "T", "data": {"label": "l", "value": 1}}
    assert "id" in validate_panels([nameless])

    assert "list" in validate_panels({"not": "a list"})
