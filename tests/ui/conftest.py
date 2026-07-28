import pytest


@pytest.fixture(autouse=True)
def ui_projects_tmp(tmp_path, monkeypatch):
    """Redirect the UI's project root to a tmp dir for every ui test, the same
    contract KC_WORKSPACE follows for the tool slices. The env var (not a
    patched function) so a subprocess spawned by the server inherits it too."""
    root = tmp_path / "projects"
    monkeypatch.setenv("KC_UI_PROJECTS", str(root))
    return root
