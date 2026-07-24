"""The knowledge slice's gated action: pull third-party data into the project.

Downloading an external dataset and merging it into training is irreversible in
the sense that matters: it changes every downstream experiment, it has
competition-rules implications (external data must be public and disclosed),
and it can silently leak the target. That is the strongest safety argument of
the three slices, and it is why this tool is irreversible=True while the other
two knowledge tools run ungated.

Everything a string comparison can decide — the ref is well-formed, the data is
not already fetched — is settled in the precheck, BEFORE the gate, so a human
is only ever asked about the one thing machines cannot judge: whether taking on
this dataset is worth its consequences.
"""

import datetime
import json
import shutil
import subprocess

from core.contracts import err, ok
from core.registry import ToolSpec
from tools.knowledge.paths import _workspace_root

_REF_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")

# Wall-clock cap on the download subprocess. External datasets for a Playground
# episode are small; anything that takes longer than this is the wrong dataset.
_DOWNLOAD_TIMEOUT_S = 600


def _external_dir():
    return _workspace_root() / "data" / "external"


def _manifest_path():
    # The disclosure record: one line per fetched dataset, kept forever. The
    # competition rules require external data to be disclosed, and this file is
    # what we disclose from.
    return _workspace_root() / "data" / "external_manifest.jsonl"


def _dest_for(dataset_ref):
    return _external_dir() / dataset_ref.split("/", 1)[1]


def _fetch_precheck(args):
    """Machine-checkable guards before the gate. Returns an error envelope or
    None. A human must never be asked to approve a ref that is not even shaped
    like a Kaggle dataset, or a download we already have."""
    ref = args["dataset_ref"]
    parts = ref.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1] or not set(ref) <= _REF_CHARS | {"/"}:
        return err(
            "bad_input",
            "dataset_ref must look like 'owner/dataset-name', got %r" % ref,
        )
    if _dest_for(ref).exists():
        # Already fetched. Re-downloading cannot be approved into a different
        # outcome, so reject here rather than burn a human decision on it.
        return err(
            "bad_input",
            "dataset already fetched at %s — use the existing copy" % _dest_for(ref),
        )
    return None


def _fetch_preview(args):
    """What the human sees at the gate. It names the consequence, not the
    mechanics: what will land where, what it changes downstream, and the
    rules obligations that come with saying yes."""
    ref = args["dataset_ref"]
    merge = args["merge_into_training"]
    lines = [
        "FETCH EXTERNAL DATASET — this changes the project, not just a file.",
        "",
        "dataset:  %s (third-party, from Kaggle)" % ref,
        "into:     %s" % _dest_for(ref),
        "merge:    %s" % (
            "YES — it will be merged into the training set, changing every "
            "experiment that runs after this point" if merge
            else "no — downloaded only, not yet part of training"
        ),
        "",
        "By approving you accept the competition-rules obligations: external",
        "data must be public and must be disclosed. It will be recorded in",
        "%s." % _manifest_path(),
        "It can also silently leak the target into training. If unsure, deny.",
    ]
    return "\n".join(lines)


def _download(dataset_ref, dest):
    """Run the actual download. The single monkeypatch point for tests, which
    replace it with a fake that populates `dest` or returns a canned error.
    Returns None on success or an error envelope."""
    try:
        proc = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_ref,
             "-p", str(dest), "--unzip"],
            capture_output=True,
            text=True,
            timeout=_DOWNLOAD_TIMEOUT_S,
        )
    except FileNotFoundError:
        return err("exec_failed", "kaggle CLI not found; install and configure it first")
    except subprocess.TimeoutExpired:
        return err(
            "timeout",
            "download exceeded %ss; pick a smaller dataset" % _DOWNLOAD_TIMEOUT_S,
            retryable=False,
        )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        if "404" in stderr:
            return err("not_found", "no such dataset on Kaggle: %r" % dataset_ref)
        return err("exec_failed", "kaggle download failed: %s" % stderr[-500:])
    return None


def fetch_external_dataset(args):
    ref = args["dataset_ref"]
    merge = args["merge_into_training"]
    dest = _dest_for(ref)

    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return err("exec_failed", "could not create %s: %s" % (dest, e))

    e = _download(ref, dest)
    if e is not None:
        # A half-written download must not survive: the precheck's
        # already-fetched rejection would otherwise see the leftover directory
        # and block the retry this failure may deserve.
        shutil.rmtree(dest, ignore_errors=True)
        return e

    row = {
        "dataset_ref": ref,
        "dest": str(dest),
        "merged_into_training": merge,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        with _manifest_path().open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        return err("exec_failed", "downloaded, but could not record disclosure: %s" % e)

    files = sorted(p.name for p in dest.iterdir())
    return ok(
        dataset_ref=ref,
        dest=str(dest),
        merged_into_training=merge,
        files=files,
    )


FETCH_EXTERNAL_DATASET = ToolSpec(
    name="fetch_external_dataset",
    description=(
        "Download a public third-party Kaggle dataset into the project and "
        "optionally mark it for merging into the training set. This is the "
        "knowledge slice's irreversible action: merged external data changes "
        "every later experiment, carries competition-rules obligations "
        "(public + disclosed), and can leak the target — so it always stops "
        "for human approval. Call it only after a technique note gives a "
        "concrete reason to want this specific dataset. Do NOT call it to "
        "search or browse (that is search_kaggle_discussions), do NOT call it "
        "for the competition's own data, and do NOT call it twice for the "
        "same ref — an already-fetched dataset is rejected before the gate."
    ),
    parameters={
        "dataset_ref": {"type": "str", "required": True},
        # The safe default: downloading for inspection is the smaller step, so
        # an omitted flag never silently rewires training.
        "merge_into_training": {"type": "bool", "default": False},
    },
    fn=fetch_external_dataset,
    irreversible=True,
    preview=_fetch_preview,
    precheck=_fetch_precheck,
)


def register(registry):
    registry.register(FETCH_EXTERNAL_DATASET)
    return registry
