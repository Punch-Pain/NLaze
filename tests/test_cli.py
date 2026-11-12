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
