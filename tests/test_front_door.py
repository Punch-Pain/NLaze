"""Front door (spec §1.1) — validity is checked to the first argument.

v2 checked only the head, so `cd docments`, `git statts`, and
`sudo r rf --n-preserve-rot` all returned PASSTHROUGH and never reached the corrector.
"""
import shutil

import pytest

from nlaze.verifier import valid_as_typed

HAS_DOCKER = shutil.which("docker") is not None
HAS_PODMAN = shutil.which("podman") is not None


@pytest.mark.parametrize("cmd", [
    "cd /tmp",
    "cd ..",
    "cd",                      # no arg -> $HOME
    "cd -",                    # previous directory
    "cd ~/",
])
def test_cd_valid(cmd):
    ok, why = valid_as_typed(cmd)
    assert ok, f"{cmd!r} should be valid: {why}"


def test_cd_missing_dir_is_invalid():
    ok, why = valid_as_typed("cd zzz_nonexistent_dir_12345")
    assert not ok, "cd to a missing directory must not passthrough"


@pytest.mark.parametrize("cmd", [
    "git status",
    "git log --oneline",
    "git commit -m 'x'",
    "git",
])
def test_git_real_subcommand_valid(cmd):
    ok, why = valid_as_typed(cmd)
    assert ok, f"{cmd!r} should be valid: {why}"


@pytest.mark.parametrize("cmd", [
    "git statts",
    "git commmit",
    "git pusch",
])
def test_git_unknown_subcommand_invalid(cmd):
    ok, why = valid_as_typed(cmd)
    assert not ok, f"{cmd!r} should be invalid: {why}"


@pytest.mark.skipif(not HAS_DOCKER, reason="docker not installed on this host")
def test_docker_real_subcommand_valid():
    ok, why = valid_as_typed("docker ps")
    assert ok, why


@pytest.mark.skipif(not HAS_PODMAN, reason="podman not installed on this host")
def test_podman_real_subcommand_valid():
    ok, why = valid_as_typed("podman ps")
    assert ok, why


@pytest.mark.parametrize("cmd", [
    "docker psss",
    "podman imgges",
])
def test_container_unknown_subcommand_invalid(cmd):
    ok, why = valid_as_typed(cmd)
    assert not ok, f"{cmd!r} should be invalid: {why}"


@pytest.mark.parametrize("cmd", [
    "ls -la",
    "rg foo src/",
    "grep -r bar .",
    "definitelynotacommand --help",
])
def test_general_head_rule(cmd):
    # head resolves -> valid; head does not -> invalid
    ok, _ = valid_as_typed(cmd)
    assert ok == (cmd.split()[0] in {"ls", "rg", "grep"}), cmd


def test_unknown_head_invalid():
    ok, why = valid_as_typed("zzz_not_a_real_command_xyz arg")
    assert not ok


@pytest.mark.parametrize("cmd", [
    "sudo git statts",
    "sudo -u root git commmit",
])
def test_sudo_unwraps_and_validates(cmd):
    ok, why = valid_as_typed(cmd)
    assert not ok, f"{cmd!r} should be invalid: {why}"


@pytest.mark.parametrize("cmd", [
    "sudo git status",
    "sudo whoami",
])
def test_sudo_real_command_valid(cmd):
    ok, why = valid_as_typed(cmd)
    assert ok, f"{cmd!r} should be valid: {why}"
