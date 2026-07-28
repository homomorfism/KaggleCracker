import pytest

from ui import projects


def test_create_project_builds_workspace_shape(ui_projects_tmp):
    meta = projects.create_project("S6E7 Health Risk", "predict health", "health_condition")
    assert meta["slug"] == "s6e7-health-risk"
    d = ui_projects_tmp / "s6e7-health-risk"
    # The directory must be shaped like workspace/ so KC_WORKSPACE can point at it.
    assert (d / "data").is_dir() and (d / "plans").is_dir() and (d / "runs").is_dir()
    assert projects.get_project("s6e7-health-risk")["target"] == "health_condition"


def test_duplicate_project_rejected_and_nothing_overwritten(ui_projects_tmp):
    projects.create_project("Demo")
    projects.save_data_file("demo", "train.csv", b"a,b\n1,2\n")
    with pytest.raises(ValueError):
        projects.create_project("Demo")
    # The consequence that matters: the existing project's data survived.
    assert (ui_projects_tmp / "demo" / "data" / "train.csv").read_bytes() == b"a,b\n1,2\n"


def test_unnameable_project_rejected(ui_projects_tmp):
    with pytest.raises(ValueError):
        projects.create_project("###")
    assert projects.list_projects() == []


def test_uploaded_file_appears_in_listing(ui_projects_tmp):
    projects.create_project("Demo")
    saved = projects.save_data_file("demo", "train.csv", b"a,b\n1,2\n")
    assert saved == {"name": "train.csv", "bytes": 8}
    files = projects.get_project("demo")["files"]
    assert files == [{"name": "train.csv", "bytes": 8}]


@pytest.mark.parametrize(
    "name",
    ["../../evil.csv", "sub/dir.csv", ".hidden.csv", "train.txt", "", "a" * 200 + ".csv"],
)
def test_unsafe_or_wrong_filenames_rejected_before_writing(ui_projects_tmp, name):
    projects.create_project("Demo")
    with pytest.raises(ValueError):
        projects.save_data_file("demo", name, b"a,b\n1,2\n")
    # Consequence: nothing landed anywhere under the project.
    assert projects.get_project("demo")["files"] == []


def test_upload_to_missing_project_is_not_found(ui_projects_tmp):
    with pytest.raises(FileNotFoundError):
        projects.save_data_file("ghost", "train.csv", b"a\n1\n")


def test_run_ids_are_claimed_uniquely(ui_projects_tmp):
    projects.create_project("Demo")
    first = projects.new_run_id("demo")
    second = projects.new_run_id("demo")
    # Same second, still distinct: mkdir is the claim.
    assert first != second
    assert projects.run_dir("demo", first).is_dir()
    assert projects.run_dir("demo", second).is_dir()


def test_plans_and_findings_surface_for_data_prep(ui_projects_tmp):
    from ui.run_analysis import run_demo_analysis

    projects.create_project("Demo", "predict y", "y")
    projects.save_data_file(
        "demo", "train.csv",
        b"id,x1,y\n1,0.5,fit\n2,0.9,sick\n3,0.4,fit\n4,1.2,sick\n5,0.6,fit\n",
    )
    run_demo_analysis("demo", projects.new_run_id("demo"), pace=0)

    plans = projects.list_plans("demo")
    assert [p["dataset"] for p in plans] == ["train.csv"]
    assert plans[0]["markdown"].startswith("# Preprocessing plan")

    flagged = projects.list_findings("demo")
    assert flagged and all(f["flagged"] for f in flagged)
    # The id column is unique per row, so cardinality must have flagged it.
    assert any(
        f["check_name"] == "cardinality" and f["column_name"] == "id" for f in flagged
    )
    # The unfiltered view is a superset: descriptive rows come back too.
    assert len(projects.list_findings("demo", flagged_only=False)) > len(flagged)


def test_findings_empty_before_any_run(ui_projects_tmp):
    projects.create_project("Fresh")
    assert projects.list_findings("fresh") == []
    assert projects.list_plans("fresh") == []


def test_bad_run_id_rejected(ui_projects_tmp):
    projects.create_project("Demo")
    with pytest.raises(ValueError):
        projects.run_dir("demo", "../escape")
    with pytest.raises(FileNotFoundError):
        projects.get_run("demo", "run-00000000-000000")
