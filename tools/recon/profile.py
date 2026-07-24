"""The recon slice's read-only tool: profile a CSV before anyone models it.

The line this file draws is the slice's whole point: DIRTY VALUES ARE DATA,
BROKEN STRUCTURE IS AN ERROR. A column mixing "N/A", "" and 3.7 is a finding
the dtypes check reports (that is exactly what a preprocessing plan needs to
know about); a file whose rows have different lengths is a bad_input error,
because no per-column statistic computed from misaligned rows can be trusted.
A naive profiler collapses the two, force-fits the rows, and hands back
confident numbers about a file it never actually read.

Everything here is stdlib (csv, math, collections) — the hard no-dependencies
rule — which also keeps every formula short enough to defend at a whiteboard.
"""

import csv
import math
from collections import Counter

from core.contracts import err, ok
from core.registry import ToolSpec
from tools.recon.paths import data_dir

# Tokens treated as a missing value everywhere in this module, compared after
# strip().lower(). One constant so missingness, dtypes and the numeric checks
# can never disagree about what "missing" means.
_MISSING_TOKENS = ("", "na", "n/a", "null", "none", "nan")

# A column counts as cleanly one type when at least this share of its present
# values parse as that type; below it the column is flagged mixed. 0.95 rather
# than 1.0 so a handful of typos flags the column instead of hiding it.
_MAJORITY = 0.95

# A column is numeric enough to summarize when at least half of its present
# values parse as numbers; below that a mean would describe a minority.
_NUMERIC_SHARE = 0.5

# Drift flags: standardized mean difference for numeric columns, total
# variation distance for categorical ones. Both on a 0-based scale where 0.1
# is a visible-but-not-alarming shift; a Playground train/test split drawn
# from one distribution should sit well under both.
_SMD_THRESHOLD = 0.1
_TVD_THRESHOLD = 0.1

# How many unseen categories to list per column before truncating: enough to
# act on, without an ID-like column flooding the report.
_MAX_UNSEEN_LISTED = 10

_TARGET_CHECKS = ("target_balance", "correlation_with_target")

_CHECKS = [
    "missingness",
    "dtypes",
    "cardinality",
    "target_balance",
    "numeric_summary",
    "correlation_with_target",
    "train_test_drift",
]


def _is_missing(value):
    return value.strip().lower() in _MISSING_TOKENS


def _try_float(value):
    try:
        return float(value)
    except ValueError:
        return None


# --- reading ----------------------------------------------------------------


def _read_table(path, sample_rows):
    """Read the header and up to sample_rows data rows.

    Returns (header, rows, None) or (None, None, error_envelope). Structural
    failures — empty file, duplicate column names, ragged rows — are bad_input
    errors here, because every downstream statistic assumes a rectangle.
    Sampling takes the FIRST sample_rows rows, deterministically: the same call
    always profiles the same bytes, which keeps runs reproducible and the
    choice explainable.
    """
    try:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return None, None, err("bad_input", "%s is empty" % path.name)
            if len(set(header)) != len(header):
                return None, None, err(
                    "bad_input",
                    "%s has duplicate column names; per-column stats would collide" % path.name,
                )
            rows = []
            # Line numbers are 1-based and the header is line 1, so data starts at 2.
            for lineno, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    return None, None, err(
                        "bad_input",
                        "ragged csv: line %d of %s has %d fields but the header has %d; "
                        "fix the file before profiling it"
                        % (lineno, path.name, len(row), len(header)),
                    )
                rows.append(row)
                if len(rows) >= sample_rows:
                    break
    except OSError as e:
        return None, None, err("exec_failed", "could not read %s: %s" % (path, e))
    if not rows:
        return None, None, err("bad_input", "%s has a header but no data rows" % path.name)
    return header, rows, None


def _columns(header, rows):
    """Transpose the rectangle into {column_name: [values]}. Safe because
    _read_table already rejected ragged rows and duplicate names."""
    cols = {name: [] for name in header}
    for row in rows:
        for name, value in zip(header, row):
            cols[name].append(value)
    return cols


def _numeric_values(values):
    """The present values that parse as numbers, as floats."""
    out = []
    for v in values:
        if _is_missing(v):
            continue
        x = _try_float(v.strip())
        if x is not None:
            out.append(x)
    return out


def _mostly_numeric(values):
    present = [v for v in values if not _is_missing(v)]
    if not present:
        return False
    return len(_numeric_values(values)) / len(present) >= _NUMERIC_SHARE


def _mean_std(nums):
    mean = sum(nums) / len(nums)
    # Population variance: we describe exactly the rows profiled, we do not
    # estimate a hidden population, so no n-1 correction.
    var = sum((x - mean) ** 2 for x in nums) / len(nums)
    return mean, math.sqrt(var)


def _quantile(sorted_vals, q):
    # Nearest-rank on purpose: no interpolation, so the answer is always a
    # value that actually occurs in the column, and the rule fits on one line.
    idx = max(0, math.ceil(q * len(sorted_vals)) - 1)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


# --- the checks ---------------------------------------------------------------


def _check_missingness(cols):
    report = {}
    for name, values in cols.items():
        n = sum(1 for v in values if _is_missing(v))
        report[name] = {"missing": n, "pct": round(n / len(values), 4)}
    return report


def _check_dtypes(cols):
    report = {}
    for name, values in cols.items():
        counts = {"int": 0, "float": 0, "text": 0, "missing": 0}
        for v in values:
            if _is_missing(v):
                counts["missing"] += 1
                continue
            s = v.strip()
            try:
                int(s)
                counts["int"] += 1
            except ValueError:
                if _try_float(s) is not None:
                    counts["float"] += 1
                else:
                    counts["text"] += 1
        present = counts["int"] + counts["float"] + counts["text"]
        if present == 0:
            majority, mixed = "missing", False
        elif counts["int"] / present >= _MAJORITY:
            majority, mixed = "int", False
        # ints are valid floats, so the two pool when deciding "numeric":
        # a column of 1, 2, 3.5 is a float column, not a mixed one.
        elif (counts["int"] + counts["float"]) / present >= _MAJORITY:
            majority, mixed = "float", False
        elif counts["text"] / present >= _MAJORITY:
            majority, mixed = "text", False
        else:
            # No type owns the column. Still report the biggest bucket, but the
            # mixed flag is the finding the plan must deal with.
            majority = max(("int", "float", "text"), key=lambda t: counts[t])
            mixed = True
        report[name] = dict(counts, majority=majority, mixed=mixed)
    return report


def _check_cardinality(cols):
    report = {}
    for name, values in cols.items():
        distinct = len({v.strip() for v in values})
        report[name] = {
            "distinct": distinct,
            # Every row unique -> probably an identifier, not a feature.
            "likely_id": distinct == len(values) and len(values) > 1,
            "constant": distinct == 1,
        }
    return report


def _check_target_balance(cols, target):
    values = cols[target]
    counts = Counter("<missing>" if _is_missing(v) else v.strip() for v in values)
    total = len(values)
    return {
        cls: {"count": n, "ratio": round(n / total, 4)}
        for cls, n in counts.most_common()
    }


def _check_numeric_summary(cols):
    report = {}
    for name, values in cols.items():
        if not _mostly_numeric(values):
            continue
        nums = sorted(_numeric_values(values))
        mean, std = _mean_std(nums)
        report[name] = {
            "count": len(nums),
            "min": nums[0],
            "max": nums[-1],
            "mean": round(mean, 6),
            "std": round(std, 6),
            "median": _quantile(nums, 0.5),
            "p5": _quantile(nums, 0.05),
            "p95": _quantile(nums, 0.95),
        }
    return report


def _pearson(xs, ys):
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None  # a constant has no direction; its correlation is undefined
    return cov / math.sqrt(vx * vy)


def _encode_target(values):
    """The target as per-row floats (None where missing), or a skip note.

    Numeric targets pass through; a two-class text target encodes as 0/1 in
    sorted class order (deterministic). Anything else is SKIPPED WITH A NOTE,
    not an error: an unhelpful target is a true fact about the data, and the
    other requested checks still ran fine.
    """
    encoded = []
    for v in values:
        if _is_missing(v):
            encoded.append(None)
            continue
        x = _try_float(v.strip())
        if x is None:
            break
        encoded.append(x)
    else:
        return encoded, None

    classes = sorted({v.strip() for v in values if not _is_missing(v)})
    if len(classes) != 2:
        return None, (
            "skipped: target has %d text classes; correlation needs a numeric "
            "or two-class target" % len(classes)
        )
    lo = classes[0]
    return [
        None if _is_missing(v) else (0.0 if v.strip() == lo else 1.0) for v in values
    ], None


def _check_correlation(cols, target):
    encoded, note = _encode_target(cols[target])
    if encoded is None:
        return {"note": note}
    report = {}
    for name, values in cols.items():
        if name == target:
            continue
        xs, ys = [], []
        # Pairwise-complete: a row counts only when both the feature and the
        # target are present and numeric on that row.
        for v, t in zip(values, encoded):
            if t is None or _is_missing(v):
                continue
            x = _try_float(v.strip())
            if x is None:
                continue
            xs.append(x)
            ys.append(t)
        if len(xs) < 2:
            continue
        r = _pearson(xs, ys)
        if r is not None:
            report[name] = round(r, 4)
    return report


def _check_drift(cols, other_cols):
    """Compare the profiled file against a second one, column by column.

    Numeric columns: standardized mean difference |m1-m2|/std1. Categorical
    (and constant-numeric, where an std of 0 makes SMD undefined): total
    variation distance between category frequencies, PLUS the categories that
    appear in the compare file but never in this one — the signal that
    silently breaks anything fitted on train categories.
    """
    shared = [c for c in cols if c in other_cols]
    columns = {}
    flagged = []
    for name in shared:
        a, b = cols[name], other_cols[name]
        if _mostly_numeric(a) and _mostly_numeric(b):
            mean_a, std_a = _mean_std(_numeric_values(a))
            mean_b, _ = _mean_std(_numeric_values(b))
            if std_a > 0:
                smd = abs(mean_a - mean_b) / std_a
                entry = {
                    "type": "numeric",
                    "mean": round(mean_a, 6),
                    "mean_compare": round(mean_b, 6),
                    "smd": round(smd, 4),
                    "drifted": smd > _SMD_THRESHOLD,
                }
                columns[name] = entry
                if entry["drifted"]:
                    flagged.append(name)
                continue
            # std 0: constant in this file -> fall through to the frequency
            # comparison, which handles "constant here, varied there" cleanly.
        p = Counter(v.strip() for v in a)
        q = Counter(v.strip() for v in b)
        pn, qn = sum(p.values()), sum(q.values())
        tvd = 0.5 * sum(
            abs(p.get(k, 0) / pn - q.get(k, 0) / qn) for k in set(p) | set(q)
        )
        unseen = sorted(set(q) - set(p))
        entry = {
            "type": "categorical",
            "tvd": round(tvd, 4),
            "unseen_in_compare": unseen[:_MAX_UNSEEN_LISTED],
            "unseen_count": len(unseen),
            "drifted": tvd > _TVD_THRESHOLD or bool(unseen),
        }
        columns[name] = entry
        if entry["drifted"]:
            flagged.append(name)
    return {
        "only_in_this_file": [c for c in cols if c not in other_cols],
        "only_in_compare": [c for c in other_cols if c not in cols],
        "columns": columns,
        "flagged": flagged,
    }


# --- precheck and body --------------------------------------------------------


def _profile_precheck(args):
    """Machine-checkable guards before the body: both paths stay inside
    workspace/data (same resolve-and-compare shape as the exec slice), and a
    check that needs an extra argument must have been given it. Returns an
    error envelope or None."""
    root = data_dir().resolve()
    for label in ("path", "compare_path"):
        ref = args.get(label)
        if ref is None:
            continue
        if not ref.strip():
            return err("bad_input", "%s is empty" % label)
        # resolve() collapses '..' and follows symlinks, and joining an absolute
        # path discards root entirely — both then land outside workspace/data.
        candidate = (data_dir() / ref).resolve()
        if candidate != root and root not in candidate.parents:
            return err("bad_input", "%s escapes workspace/data: %r" % (label, ref))

    checks = args["checks"]
    if not checks:
        return err("bad_input", "checks is empty; pick at least one check to run")
    needy = [c for c in checks if c in _TARGET_CHECKS]
    if needy and "target" not in args:
        return err(
            "bad_input",
            "%s need the target parameter (the column being predicted)" % ", ".join(needy),
        )
    if "train_test_drift" in checks and "compare_path" not in args:
        return err(
            "bad_input",
            "train_test_drift needs compare_path (the file to compare against, e.g. test.csv)",
        )
    return None


def profile_dataset(args):
    path = data_dir() / args["path"]
    # is_file(), not exists(): a directory is just as un-profilable as nothing.
    if not path.is_file():
        return err("not_found", "dataset not found under workspace/data: %r" % args["path"])

    header, rows, e = _read_table(path, args["sample_rows"])
    if e is not None:
        return e
    cols = _columns(header, rows)

    checks = args["checks"]
    target = args.get("target")
    # Presence of the target parameter was settled in the precheck; whether the
    # named column exists requires reading the file, so it is settled here.
    if any(c in _TARGET_CHECKS for c in checks) and target not in cols:
        return err(
            "bad_input",
            "target column %r not in %s; columns are %r" % (target, path.name, header),
        )

    other_cols = None
    if "train_test_drift" in checks:
        compare = data_dir() / args["compare_path"]
        if not compare.is_file():
            return err(
                "not_found",
                "compare file not found under workspace/data: %r" % args["compare_path"],
            )
        c_header, c_rows, e = _read_table(compare, args["sample_rows"])
        if e is not None:
            return e
        other_cols = _columns(c_header, c_rows)

    report = {"path": args["path"], "rows_profiled": len(rows), "columns": header}
    for check in checks:
        if check == "missingness":
            report[check] = _check_missingness(cols)
        elif check == "dtypes":
            report[check] = _check_dtypes(cols)
        elif check == "cardinality":
            report[check] = _check_cardinality(cols)
        elif check == "target_balance":
            report[check] = _check_target_balance(cols, target)
        elif check == "numeric_summary":
            report[check] = _check_numeric_summary(cols)
        elif check == "correlation_with_target":
            report[check] = _check_correlation(cols, target)
        elif check == "train_test_drift":
            report[check] = _check_drift(cols, other_cols)
    return ok(**report)


PROFILE_DATASET = ToolSpec(
    name="profile_dataset",
    description=(
        "Compute per-column statistics for a CSV under workspace/data and "
        "return a structured report: missingness, dtypes, cardinality, target "
        "balance, numeric summaries, correlation with the target, and train/"
        "test drift. Call it to understand a dataset BEFORE writing its "
        "preprocessing plan. Do NOT call it to clean or transform data — it "
        "only reads. Do NOT re-profile a file with the same checks in one run; "
        "reuse the earlier report from the transcript. Do NOT point it at "
        "anything outside workspace/data."
    ),
    parameters={
        "path": {"type": "str", "required": True},
        # Constrained AND required: the model must say which questions it is
        # asking; "profile everything" hides what the plan is grounded on.
        "checks": {"type": "list", "required": True, "item_enum": _CHECKS},
        "target": {"type": "str"},
        "compare_path": {"type": "str"},
        # Capped in the schema so an oversized ask is a bad_input rejection
        # before a single row is read.
        "sample_rows": {"type": "int", "min": 100, "max": 200_000, "default": 100_000},
    },
    fn=profile_dataset,
    precheck=_profile_precheck,
    # Reversible: read-only, writes nothing anywhere, so it runs ungated. The
    # contrast with overwrite_preprocessing_plan is the point of requirement 3.
)


def register(registry):
    registry.register(PROFILE_DATASET)
    return registry
