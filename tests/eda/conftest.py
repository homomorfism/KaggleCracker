import pytest


@pytest.fixture(autouse=True)
def eda_workspace(tmp_path, monkeypatch):
    """Every eda test runs against a tmp workspace, mirroring the KC_WORKSPACE
    contract the other tool slices already follow."""
    ws = tmp_path / "workspace"
    (ws / "data").mkdir(parents=True)
    monkeypatch.setenv("KC_WORKSPACE", str(ws))
    return ws
