"""Command line entry points.

    kagglecracker preflight     # verify everything the worker needs, before it needs it
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

from kagglecracker.config import Settings, load_settings
from kagglecracker.runtime.pricing import UnpricedModelError, assert_all_priced

app = typer.Typer(add_completion=False, help="Autonomous Kaggle tabular competition agent.")

MIN_KAGGLE_CLI = (2, 2, 0)


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""

    @property
    def blocking(self) -> bool:
        return not self.ok and bool(self.fix)


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{' '.join(cmd)}: timed out after {timeout}s"
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _parse_version(text: str) -> tuple[int, ...]:
    """Pull the first dotted-number run out of a version banner."""
    import re

    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not m:
        return ()
    return tuple(int(g) for g in m.groups() if g is not None)


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def check_kaggle_cli() -> Check:
    if shutil.which("kaggle") is None:
        return Check(
            "kaggle CLI",
            False,
            "not on PATH",
            "uv sync  (the `kaggle` package is a project dependency)",
        )
    code, out, err = _run(["kaggle", "--version"])
    if code != 0:
        return Check("kaggle CLI", False, err or "non-zero exit", "Reinstall: uv sync")
    version = _parse_version(out)
    if version < MIN_KAGGLE_CLI:
        want = ".".join(str(n) for n in MIN_KAGGLE_CLI)
        return Check(
            "kaggle CLI",
            False,
            f"{out} — need >= {want} for the `forums` commands",
            f"uv pip install -U 'kaggle>={want}'",
        )
    return Check("kaggle CLI", True, out)


def check_kaggle_auth() -> Check:
    """`competitions list` proves credentials AND network in one call."""
    code, out, err = _run(["kaggle", "competitions", "list", "--csv"], timeout=90)
    if code != 0:
        blob = (err + out).lower()
        if "401" in blob or "unauthor" in blob or "credential" in blob:
            fix = (
                "Create an API token at kaggle.com/settings -> API -> Create New Token, "
                "then save it to ~/.kaggle/kaggle.json (chmod 600)."
            )
        else:
            fix = "Check network reachability to kaggle.com."
        lines = (err or out).splitlines()
        return Check("kaggle auth", False, lines[-1] if lines else "failed", fix)
    n = max(0, len(out.splitlines()) - 1)
    return Check("kaggle auth", True, f"authenticated, {n} competitions listed")


def check_competition(settings: Settings) -> Check:
    """A successful file listing proves the competition rules were accepted.

    Kaggle refuses data access until you accept the rules in a browser. There is
    no API for that acceptance, so this check can only report it, never fix it.
    """
    if not settings.competition:
        return Check(
            "competition",
            False,
            "not set",
            "Pick a tabular playground competition and set KC_COMPETITION=<slug>.",
        )
    code, out, err = _run(
        ["kaggle", "competitions", "files", settings.competition, "--csv"], timeout=90
    )
    blob = (err + out).lower()
    if code != 0 or "403" in blob or "rules" in blob:
        return Check(
            "competition",
            False,
            f"{settings.competition}: cannot list files",
            f"Open https://www.kaggle.com/c/{settings.competition}/rules and click "
            "'I Understand and Accept'. There is no API for this.",
        )
    return Check("competition", True, f"{settings.competition}: rules accepted, files listable")


def check_docker(settings: Settings) -> Check:
    if shutil.which("docker") is None:
        return Check("docker", False, "not on PATH", "Install Docker Desktop.")
    code, out, err = _run(
        ["docker", "info", "--format", "{{.OSType}} {{.NCPU}} {{.MemTotal}}"], timeout=30
    )
    if code != 0:
        return Check("docker", False, "daemon not reachable", "Start Docker Desktop.")
    parts = out.split()
    if len(parts) != 3:
        return Check("docker", True, out)
    ostype, ncpu_s, mem_s = parts
    ncpu, mem_bytes = int(ncpu_s), int(mem_s)
    mem_gb = mem_bytes / 1024**3
    # Docker Desktop caps --cpus/--memory at the VM allocation. Asking for more
    # than the VM has does not error; it silently under-delivers, and a node that
    # should finish in 300s hits the 900s timeout instead.
    want_mem_gb = float(settings.docker_memory.rstrip("gG"))
    if ncpu < settings.docker_cpus or mem_gb < want_mem_gb:
        return Check(
            "docker",
            False,
            f"VM has {ncpu} CPU / {mem_gb:.1f} GB; config wants "
            f"{settings.docker_cpus} CPU / {want_mem_gb:.0f} GB",
            "Raise the VM allocation in Docker Desktop > Settings > Resources, "
            "or lower KC_DOCKER_CPUS / KC_DOCKER_MEMORY.",
        )
    return Check("docker", True, f"{ostype}, {ncpu} CPU, {mem_gb:.1f} GB available")


def check_sandbox_image(settings: Settings) -> Check:
    code, out, _ = _run(["docker", "image", "inspect", settings.docker_image], timeout=30)
    if code != 0:
        return Check(
            "sandbox image",
            False,
            f"{settings.docker_image} not built",
            f"docker build -t {settings.docker_image} sandbox/",
        )
    try:
        size_gb = json.loads(out)[0]["Size"] / 1024**3
        return Check("sandbox image", True, f"{settings.docker_image} ({size_gb:.1f} GB)")
    except (json.JSONDecodeError, KeyError, IndexError):
        return Check("sandbox image", True, settings.docker_image)


def check_llm_key(settings: Settings) -> Check:
    if settings.api_key:
        return Check("llm key", True, f"{settings.provider} key present")
    env_name = f"{settings.provider.upper()}_API_KEY"
    return Check("llm key", False, f"{env_name} unset", f"export {env_name}=...")


def check_pricing(settings: Settings) -> Check:
    models = settings.models_in_use()
    try:
        assert_all_priced(settings.provider, models)
    except UnpricedModelError as exc:
        return Check("pricing", False, str(exc).splitlines()[0], str(exc).splitlines()[-1])
    return Check("pricing", True, f"{len(models)} model(s) priced: {', '.join(sorted(models))}")


def check_data(settings: Settings) -> Check:
    required = ("train.csv", "test.csv", "sample_submission.csv")
    if not settings.data_dir.is_dir():
        return Check("data", False, f"{settings.data_dir} missing", "kagglecracker download")
    present = {p.name for p in settings.data_dir.iterdir()}
    missing = [f for f in required if f not in present]
    if missing:
        return Check(
            "data",
            False,
            f"missing {', '.join(missing)} in {settings.data_dir}",
            "kagglecracker download",
        )
    return Check("data", True, f"{len(present)} file(s) in {settings.data_dir}")


# --------------------------------------------------------------------------


@app.command()
def preflight() -> None:
    """Verify every external dependency before anything depends on it."""
    settings = load_settings()
    checks = [
        check_kaggle_cli(),
        check_kaggle_auth(),
        check_competition(settings),
        check_data(settings),
        check_docker(settings),
        check_sandbox_image(settings),
        check_llm_key(settings),
        check_pricing(settings),
    ]

    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        typer.echo(f"  [{mark}] {c.name.ljust(width)}  {c.detail}")
        if not c.ok and c.fix:
            typer.echo(f"         {' ' * width}  -> {c.fix}")

    failed = [c for c in checks if not c.ok]
    typer.echo("")
    if failed:
        typer.echo(f"{len(failed)}/{len(checks)} checks failed.")
        raise typer.Exit(code=1)
    typer.echo(f"All {len(checks)} checks passed.")


@app.command()
def download() -> None:
    """Download and unpack the competition data into `data/`.

    kaggle-cli 2.x dropped the `--unzip` flag its 1.x predecessor had, so the
    archive comes down zipped and we expand it here.
    """
    import zipfile

    settings = load_settings()
    if not settings.competition:
        typer.echo("KC_COMPETITION is not set.", err=True)
        raise typer.Exit(code=1)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle", "competitions", "download",
        settings.competition, "-p", str(settings.data_dir), "-o",
    ]
    typer.echo(f"$ {' '.join(cmd)}")
    code, out, err = _run(cmd, timeout=1800)
    if out:
        typer.echo(out)
    if code != 0:
        typer.echo(err, err=True)
        raise typer.Exit(code=code)

    for archive in settings.data_dir.glob("*.zip"):
        typer.echo(f"unzip {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(settings.data_dir)
        archive.unlink()

    files = sorted(p.name for p in settings.data_dir.iterdir())
    typer.echo(f"{len(files)} file(s): {', '.join(files)}")


@app.command()
def init() -> None:
    """Create the database and seed the CV protocol for the configured competition."""
    from kagglecracker.db.store import ProtocolStore, init_db

    settings = load_settings()
    conn = init_db(settings.db_path)
    store = ProtocolStore(conn)

    existing = store.current()
    if existing:
        typer.echo(f"Protocol already set: {existing.describe()}")
        typer.echo(f"  rationale: {existing.rationale}")
        return

    if settings.competition != "spaceship-titanic":
        typer.echo(
            f"No seed protocol defined for '{settings.competition}'. Read the "
            "competition's evaluation page and add one — do not guess the metric.",
            err=True,
        )
        raise typer.Exit(code=1)

    protocol = store.set_current(
        metric_name="accuracy",
        direction="higher_is_better",
        fold_scheme="StratifiedGroupKFold",
        n_folds=5,
        seed=42,
        fold_params={"group_key": "PassengerId.split('_')[0]", "shuffle": True},
        rationale=(
            "Metric is classification accuracy per the competition evaluation page; the "
            "target is near-balanced (50.4/49.6) so accuracy is well-behaved. Grouping is "
            "NOT optional: PassengerId is 'gggg_pp' where gggg is the travelling party, and "
            "0 of 3063 test groups appear in train — Kaggle split by group. Within a party "
            "the outcome is correlated (P(transported | all mates transported) = 0.616 vs a "
            "0.504 base rate), so a plain KFold lets a model read a groupmate's label, which "
            "does not exist at test time. Stratify on the target as well to keep folds "
            "balanced. Established Day 0 from the data, not deferred to the EDA agent."
        ),
    )
    typer.echo(f"Seeded protocol #{protocol.id}: {protocol.describe()}")


def exec_(  # registered as `exec` below; `exec` is a Python keyword so it cannot be the name
    file: str = typer.Argument(..., help="Python file to run inside the sandbox"),
    node_id: str = typer.Option("manual", help="Names the run directory under runs/"),
    timeout: int = typer.Option(0, help="Wall-clock limit; 0 uses the configured default"),
) -> None:
    """Run a local Python file through the sandbox and validate its output.

    The same path a generated node takes, minus the LLM. Useful for proving the
    contract on Day 1 and for debugging a node's code by hand afterwards.
    """
    from kagglecracker.db.store import ProtocolStore, connect
    from kagglecracker.engine.contract import ContractError, SubmissionValidator, parse_metrics
    from kagglecracker.executors.base import RunStatus
    from kagglecracker.executors.local_docker import LocalDockerExecutor

    settings = load_settings()
    code = Path(file).read_text()
    executor = LocalDockerExecutor(settings)

    protocol = None
    if settings.db_path.exists():
        protocol = ProtocolStore(connect(settings.db_path, read_only=True)).current()

    typer.echo(f"running {file} in {settings.docker_image} ...")
    result = executor.run(code, timeout_s=timeout or settings.node_timeout_s, node_id=node_id)

    typer.echo(f"  status    {result.status}  ({result.duration_s:.1f}s, exit={result.exit_code})")
    typer.echo(f"  workspace {result.workspace}")
    typer.echo(f"  network   {'REACHABLE — BUG' if result.had_network else 'none (verified)'}")
    if result.stdout_tail:
        typer.echo("  --- stdout tail ---")
        for line in result.stdout_tail.splitlines()[-15:]:
            typer.echo(f"  {line}")

    if result.status is not RunStatus.OK:
        typer.echo(f"\n{result.status}: {result.error_text[:2000]}", err=True)
        raise typer.Exit(code=1)

    try:
        metrics = parse_metrics(
            result.workspace, expected_folds=protocol.n_folds if protocol else None
        )
        validator = SubmissionValidator(settings.data_dir / "sample_submission.csv")
        submission = validator.validate(result.workspace)
    except ContractError as exc:
        typer.echo(f"\ncontract_violation: {exc}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(
        f"\n  cv_score  {metrics.cv_score:.5f}"
        f"  (fold mean {metrics.fold_mean:.5f}, std {metrics.fold_std:.5f}, "
        f"min {metrics.fold_min:.5f})"
    )
    typer.echo(f"  folds     {[round(s, 5) for s in metrics.fold_scores]}")
    typer.echo(f"  submission {len(submission)} rows, contract OK")


# `exec` is a Python keyword, so the function is exec_ and the command is renamed.
app.command(name="exec")(exec_)


@app.command()
def node(
    action: str = typer.Option("draft", help="draft | improve | debug"),
    parent: int = typer.Option(0, help="Parent node id, required for improve/debug"),
    dry_run: bool = typer.Option(False, help="Print the prompt and exit without calling the LLM"),
) -> None:
    """Generate ONE node with the LLM, run it in the sandbox, score it, persist it.

    The whole per-node path, driven by hand. The Day-3 search loop is this in a
    `while`, with a policy choosing the action and budgets deciding when to stop.
    """
    from kagglecracker.db.store import NodeStore, ProtocolStore, RunStore, connect
    from kagglecracker.engine.contract import ContractError, SubmissionValidator, parse_metrics
    from kagglecracker.engine.prompts import PromptBuilder
    from kagglecracker.executors.base import RunStatus
    from kagglecracker.executors.local_docker import LocalDockerExecutor
    from kagglecracker.runtime.factory import build_runtime

    settings = load_settings()
    conn = connect(settings.db_path)
    protocol = ProtocolStore(conn).current()
    if protocol is None:
        typer.echo("No CV protocol seeded. Run `kagglecracker init` first.", err=True)
        raise typer.Exit(code=1)

    nodes_store = NodeStore(conn)
    history = nodes_store.for_protocol(protocol.id, limit=settings.tree_summary_max_nodes)
    parent_node = nodes_store.get(parent) if parent else None

    builder = PromptBuilder(settings, protocol)
    prompt = builder.render(action, nodes=history, parent=parent_node)

    typer.echo(
        f"prompt: {len(prompt)} chars, {len(history)} prior node(s) "
        f"on protocol {protocol.id}"
    )
    if dry_run:
        typer.echo("\n" + "=" * 78 + "\n" + prompt)
        return

    runtime = build_runtime(settings)
    typer.echo(f"generating with {settings.provider}/{settings.model_for(action)} ...")
    gen = runtime.generate(prompt, action=action)

    typer.echo(
        f"  {gen.status}  in={gen.usage.tokens_in} out={gen.usage.tokens_out} "
        f"cached={gen.usage.cached_in}  ${gen.usd:.4f}  stop={gen.stop_reason}"
    )
    if not gen.ok:
        typer.echo(f"\n{gen.status}: {gen.error_text}", err=True)
        raise typer.Exit(code=1)

    run_id = RunStore(conn).create(protocol.id, settings)
    executor = LocalDockerExecutor(settings)
    node_id = f"r{run_id}-{action}"
    typer.echo(f"running in sandbox (timeout {settings.node_timeout_s}s) ...")
    result = executor.run(gen.code, timeout_s=settings.node_timeout_s, node_id=node_id)
    typer.echo(f"  {result.status}  {result.duration_s:.1f}s  exit={result.exit_code}")

    status = str(result.status)
    error_text = result.error_text
    cv_score = None
    fold_scores: list[float] = []
    metrics_extra = None

    if result.status is RunStatus.OK:
        try:
            metrics = parse_metrics(result.workspace, expected_folds=protocol.n_folds)
            SubmissionValidator(settings.data_dir / "sample_submission.csv").validate(
                result.workspace
            )
            cv_score = metrics.cv_score
            fold_scores = metrics.fold_scores
            metrics_extra = metrics.extra
            error_text = ""
        except ContractError as exc:
            status, error_text = "contract_violation", str(exc)

    stored = nodes_store.insert(
        run_id=run_id,
        protocol_id=protocol.id,
        parent_id=parent or None,
        action=action,
        depth=(parent_node.depth + 1) if parent_node else 0,
        code=gen.code,
        status=status,
        error_text=error_text or None,
        cv_score=cv_score,
        fold_scores=fold_scores or None,
        metrics=metrics_extra,
        stdout_path=str(result.stdout_path) if result.stdout_path else None,
        run_flags=result.run_flags,
        prompt=prompt,
        prompt_chars=gen.prompt_chars,
        tokens_in=gen.usage.tokens_in,
        tokens_out=gen.usage.tokens_out,
        usd=gen.usd,
    )
    RunStore(conn).record_node(
        run_id, usd=gen.usd, wall_clock_s=result.duration_s, status=status
    )

    typer.echo(f"\nnode {stored.id}: {stored.status}")
    if stored.scored:
        typer.echo(f"  cv_score {stored.cv_score:.5f}  folds {[round(s, 5) for s in fold_scores]}")
    else:
        typer.echo(f"  {(error_text or '')[:600]}")
    typer.echo(f"  cost ${gen.usd:.4f}  ({gen.usage.tokens_in} in / {gen.usage.tokens_out} out)")


@app.command()
def worker(
    resume: int = typer.Option(0, help="Resume an existing run id instead of starting a new one"),
    max_nodes: int = typer.Option(0, help="Override KC_MAX_NODES for this run"),
    max_usd: float = typer.Option(0.0, help="Override KC_MAX_USD for this run"),
) -> None:
    """Run the search loop until a budget or the circuit breaker stops it.

    Sole writer. A second worker fails loudly rather than forking the tree.
    """
    from kagglecracker.db.store import LockHeld, ProtocolStore, connect
    from kagglecracker.engine.loop import SearchLoop
    from kagglecracker.engine.policy import SearchPolicy
    from kagglecracker.engine.ranking import leaderboard
    from kagglecracker.executors.local_docker import LocalDockerExecutor
    from kagglecracker.runtime.factory import build_runtime

    settings = load_settings()
    if max_nodes:
        settings.max_nodes = max_nodes
    if max_usd:
        settings.max_usd = max_usd

    conn = connect(settings.db_path)
    protocol = ProtocolStore(conn).current()
    if protocol is None:
        typer.echo("No CV protocol seeded. Run `kagglecracker init` first.", err=True)
        raise typer.Exit(code=1)

    loop = SearchLoop(
        settings=settings,
        conn=conn,
        protocol=protocol,
        runtime=build_runtime(settings),
        executor=LocalDockerExecutor(settings),
        policy=SearchPolicy(
            protocol,
            seed=protocol.seed,
            max_depth=settings.policy_max_depth,
            seed_drafts=settings.policy_seed_drafts,
        ),
        on_event=typer.echo,
    )

    typer.echo(
        f"budgets: {settings.max_nodes} nodes / {settings.max_wall_clock_s}s / "
        f"${settings.max_usd:.2f}  |  breaker: {settings.max_consecutive_failures} "
        f"failures, {settings.max_consecutive_infra_errors} infra"
    )
    try:
        result = loop.run(resume_run_id=resume or None)
    except LockHeld as exc:
        typer.echo(f"\n{exc}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(
        f"\nrun {result.run_id}: {len(result.steps)} step(s), "
        f"{result.nodes_scored} scored, ${result.usd_spent:.4f}"
    )
    board = leaderboard(conn, protocol, limit=5)
    if board:
        typer.echo("\nleaderboard (mean / std / min across folds):")
        for r in board:
            typer.echo(
                f"  node {r.node.id:>3}  {r.node.cv_score:.5f}   "
                f"mean {r.fold_mean:.5f}  std {r.fold_std:.5f}  min {r.fold_min:.5f}"
            )


@app.command()
def eda() -> None:
    """Inspect the data in the sandbox and write context/eda_findings.md.

    The script is generic and mechanical; the interpretation is the agent's.
    """
    from kagglecracker.executors.local_docker import LocalDockerExecutor
    from kagglecracker.feeders.eda import run_eda
    from kagglecracker.runtime.factory import build_runtime

    settings = load_settings()
    typer.echo("running EDA script in the sandbox ...")
    result = run_eda(
        settings=settings,
        executor=LocalDockerExecutor(settings),
        runtime=build_runtime(settings),
    )
    if not result.ok:
        typer.echo(f"\n{result.error}", err=True)
        raise typer.Exit(code=1)

    gen = result.generation
    typer.echo(f"  raw statistics: {len(result.raw_output)} chars")
    typer.echo(
        f"  summarised: in={gen.usage.tokens_in} out={gen.usage.tokens_out} ${gen.usd:.4f}"
    )
    typer.echo(f"  wrote {result.findings_path} "
               f"({len(result.findings_path.read_text())} chars)")



@app.command()
def discussions(limit: int = typer.Option(15, help="How many topics to read")) -> None:
    """Read the competition forum and write context/discussions/<date>.md."""
    from kagglecracker.feeders.discussions import run_discussions
    from kagglecracker.runtime.factory import build_runtime

    settings = load_settings()
    typer.echo(f"reading up to {limit} topics from {settings.competition} ...")
    result = run_discussions(
        settings=settings, runtime=build_runtime(settings), limit=limit
    )
    if not result.ok:
        typer.echo(f"\n{result.error}", err=True)
        raise typer.Exit(code=1)

    gen = result.generation
    typer.echo(f"  {result.topics_found} topics, {result.raw_chars} chars fetched")
    typer.echo(
        f"  summarised: in={gen.usage.tokens_in} out={gen.usage.tokens_out} ${gen.usd:.4f}"
    )
    typer.echo(f"  wrote {result.path} ({len(result.path.read_text())} chars)")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
) -> None:
    """Serve the dashboard. Reads the tree; the worker keeps writing it."""
    import uvicorn

    settings = load_settings()
    if not settings.db_path.exists():
        typer.echo(f"No database at {settings.db_path}. Run `kagglecracker init`.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"http://{host}:{port}  (reading {settings.db_path})")
    uvicorn.run(
        "kagglecracker.web.app:app", host=host, port=port, reload=reload, log_level="warning"
    )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
