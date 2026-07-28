"""The dashboard panel schema: the contract between EDA scripts and the UI.

A panel is a plain dict {id, type, title, commentary, data}. The caps here are
UI-protective: an EDA script must aggregate (bin, count, sample) before it
emits — raw rows never cross this boundary, so the dashboard file stays small
enough to poll and the charts stay renderable.

validate_panels is a pure function returning an error string or None, so the
tool body, the runner, and the tests all judge panels with the same sentence.
"""

PANEL_TYPES = ("histogram", "bar", "scatter", "heatmap", "line", "table", "stat", "markdown")

MAX_PANELS = 24
MAX_HIST_BINS = 200
MAX_BAR_CATEGORIES = 100
MAX_SCATTER_POINTS = 2000
MAX_HEATMAP_DIM = 50
MAX_TABLE_ROWS = 100


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_histogram(data):
    bins = data.get("bins")
    if not isinstance(bins, list) or not bins:
        return "histogram needs a non-empty 'bins' list"
    if len(bins) > MAX_HIST_BINS:
        return "histogram has %d bins (max %d) — use coarser bins" % (len(bins), MAX_HIST_BINS)
    for b in bins:
        if not isinstance(b, dict) or not all(_is_num(b.get(k)) for k in ("x0", "x1", "count")):
            return "each histogram bin must be {x0, x1, count} with numbers"
    return None


def _check_bar(data):
    cats = data.get("categories")
    series = data.get("series")
    if not isinstance(cats, list) or not cats:
        return "bar needs a non-empty 'categories' list"
    if len(cats) > MAX_BAR_CATEGORIES:
        return "bar has %d categories (max %d) — keep the top ones" % (len(cats), MAX_BAR_CATEGORIES)
    if not isinstance(series, list) or not series:
        return "bar needs a non-empty 'series' list"
    for s in series:
        if not isinstance(s, dict) or not isinstance(s.get("name"), str):
            return "each bar series must be {name, values}"
        values = s.get("values")
        if not isinstance(values, list) or len(values) != len(cats):
            return "series %r values must match categories length %d" % (s.get("name"), len(cats))
        if not all(_is_num(v) for v in values):
            return "series %r has non-numeric values" % s.get("name")
    return None


def _check_scatter(data):
    points = data.get("points")
    if not isinstance(points, list) or not points:
        return "scatter needs a non-empty 'points' list"
    if len(points) > MAX_SCATTER_POINTS:
        return "scatter has %d points (max %d) — downsample in the script" % (
            len(points),
            MAX_SCATTER_POINTS,
        )
    for p in points:
        if not isinstance(p, list) or len(p) < 2 or not (_is_num(p[0]) and _is_num(p[1])):
            return "each scatter point must be [x, y] numbers (optional third label)"
    return None


def _check_heatmap(data):
    xs, ys, values = data.get("x_labels"), data.get("y_labels"), data.get("values")
    if not isinstance(xs, list) or not isinstance(ys, list) or not xs or not ys:
        return "heatmap needs non-empty 'x_labels' and 'y_labels'"
    if len(xs) > MAX_HEATMAP_DIM or len(ys) > MAX_HEATMAP_DIM:
        return "heatmap is %dx%d (max %dx%d)" % (len(xs), len(ys), MAX_HEATMAP_DIM, MAX_HEATMAP_DIM)
    if not isinstance(values, list) or len(values) != len(ys):
        return "heatmap 'values' must have one row per y_label"
    for row in values:
        if not isinstance(row, list) or len(row) != len(xs):
            return "each heatmap row must have one value per x_label"
        if not all(_is_num(v) or v is None for v in row):
            return "heatmap values must be numbers (or null for missing)"
    return None


def _check_line(data):
    xs, series = data.get("x"), data.get("series")
    if not isinstance(xs, list) or not xs:
        return "line needs a non-empty 'x' list"
    if not isinstance(series, list) or not series:
        return "line needs a non-empty 'series' list"
    for s in series:
        values = s.get("values") if isinstance(s, dict) else None
        if not isinstance(s, dict) or not isinstance(s.get("name"), str):
            return "each line series must be {name, values}"
        if not isinstance(values, list) or len(values) != len(xs):
            return "series %r values must match x length %d" % (s.get("name"), len(xs))
    return None


def _check_table(data):
    cols, rows = data.get("columns"), data.get("rows")
    if not isinstance(cols, list) or not cols:
        return "table needs a non-empty 'columns' list"
    if not isinstance(rows, list):
        return "table needs a 'rows' list"
    if len(rows) > MAX_TABLE_ROWS:
        return "table has %d rows (max %d) — aggregate first" % (len(rows), MAX_TABLE_ROWS)
    for row in rows:
        if not isinstance(row, list) or len(row) != len(cols):
            return "each table row must have one cell per column"
    return None


def _check_stat(data):
    if not isinstance(data.get("label"), str) or "value" not in data:
        return "stat needs 'label' and 'value'"
    return None


def _check_markdown(data):
    if not isinstance(data.get("text"), str) or not data["text"].strip():
        return "markdown needs non-empty 'text'"
    return None


_CHECKS = {
    "histogram": _check_histogram,
    "bar": _check_bar,
    "scatter": _check_scatter,
    "heatmap": _check_heatmap,
    "line": _check_line,
    "table": _check_table,
    "stat": _check_stat,
    "markdown": _check_markdown,
}


def validate_panels(panels):
    """Error string naming the offending panel, or None if all panels are valid."""
    if not isinstance(panels, list):
        return "expected a JSON list of panel objects"
    if len(panels) > MAX_PANELS:
        return "%d panels (max %d)" % (len(panels), MAX_PANELS)
    seen = set()
    for i, panel in enumerate(panels):
        where = "panel %d" % i
        if not isinstance(panel, dict):
            return "%s: not an object" % where
        pid = panel.get("id")
        if not isinstance(pid, str) or not pid.strip():
            return "%s: missing string 'id'" % where
        where = "panel %r" % pid
        if pid in seen:
            return "%s: duplicate id" % where
        seen.add(pid)
        ptype = panel.get("type")
        if ptype not in PANEL_TYPES:
            return "%s: type must be one of %s, got %r" % (where, "/".join(PANEL_TYPES), ptype)
        if not isinstance(panel.get("title"), str) or not panel["title"].strip():
            return "%s: missing string 'title'" % where
        if not isinstance(panel.get("commentary", ""), str):
            return "%s: 'commentary' must be a string" % where
        data = panel.get("data")
        if not isinstance(data, dict):
            return "%s: missing object 'data'" % where
        problem = _CHECKS[ptype](data)
        if problem:
            return "%s: %s" % (where, problem)
    return None
