import json, sys
import pytest
from nlaze import cli

@pytest.mark.parametrize("line,expected_lane", [
    ("ls -la", "passthrough"),
    ("sudo whoami", "danger_mild"),
    ("rm -rf / --no-preserve-root", "danger_high"),
    ("nonsense_xyz_abc", "confirm"),
])
def test_cli_check_json(capsys, line, expected_lane):
    old_argv = sys.argv
    sys.argv = ["nlaze", "--json", "check", line]
    try:
        rc = cli.main()
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["lane"] == expected_lane
    finally:
        sys.argv = old_argv


def _run(tmp_path, monkeypatch, args):
    monkeypatch.setenv("NLAZE_POLICY_PATH", str(tmp_path / "p.sqlite"))
    old_argv = sys.argv
    sys.argv = ["nlaze"] + args
    try:
        rc = cli.main()
    finally:
        sys.argv = old_argv
    return rc


def test_cli_allowlist_add_list_remove(tmp_path, capsys, monkeypatch):
    assert _run(tmp_path, monkeypatch, ["allowlist", "add", "opencode"]) == 0
    assert "opencode" in capsys.readouterr().out

    assert _run(tmp_path, monkeypatch, ["allowlist", "add", "opencode", "--flag=--remove-database"]) == 0
    capsys.readouterr()
    assert _run(tmp_path, monkeypatch, ["allowlist", "list"]) == 0
    out = capsys.readouterr().out
    assert "opencode" in out
    assert "--remove-database" in out

    assert _run(tmp_path, monkeypatch, ["allowlist", "remove", "opencode", "--flag=--remove-database"]) == 0
    capsys.readouterr()
    assert _run(tmp_path, monkeypatch, ["allowlist", "remove", "opencode"]) == 0
    capsys.readouterr()
    assert _run(tmp_path, monkeypatch, ["allowlist", "list"]) == 0
    out = capsys.readouterr().out
    assert "opencode" not in out


def test_cli_danger_add_list_remove(tmp_path, capsys, monkeypatch):
    assert _run(tmp_path, monkeypatch, ["danger", "add", "opencode", "--flag=--drop-data"]) == 0
    capsys.readouterr()
    assert _run(tmp_path, monkeypatch, ["danger", "list"]) == 0
    assert "--drop-data" in capsys.readouterr().out

    assert _run(tmp_path, monkeypatch, ["danger", "remove", "opencode", "--flag=--drop-data"]) == 0
    capsys.readouterr()
    assert _run(tmp_path, monkeypatch, ["danger", "list"]) == 0
    assert "--drop-data" not in capsys.readouterr().out
