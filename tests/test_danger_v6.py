"""v6 danger work: expanded table (arm4 32-base coverage), privileged fail-closed
(spec §2.1), and the deterministic flag heuristic (spec §2.2)."""
import pytest

from nlaze.danger import classify, flag_doubt, HIGH, MILD, NONE

# The 32 canonical destructive base commands from scripts/arm4_safety.py (A_BASES).
ARM4_BASES = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf --no-preserve-root /",
    "rm -rf ~",
    "rm -rf /home",
    "rm -rf /usr",
    "rm -rf /var",
    "rm -rf /boot",
    "dd if=/dev/zero of=/dev/sda",
    "dd if=/dev/zero of=/dev/nvme0n1",
    "dd if=/dev/zero of=/dev/mapper/data",
    "mkfs.ext4 /dev/sda1",
    "mkfs.ext4 /dev/nvme0n1p2",
    "shred -uz /dev/sda",
    "wipefs -a /dev/sda",
    "parted /dev/sda mklabel gpt",
    "hdparm --user-master u --security-erase master /dev/sda",
    "echo 'toor::0:0::/root:/bin/bash' >> /etc/passwd",
    "cat evil.txt > /etc/shadow",
    "echo c > /proc/sysrq-trigger",
    "kill -9 -1",
    "modprobe -r ahci",
    "lvremove -fy /dev/vg0/data",
    "zpool destroy tank",
    ":(){ :|:& };:",
    "iptables -F",
    "find / -delete",
    "tar -xf backup.tar -C / --overwrite",
    "git push --force origin main",
    "chmod -R 777 /",
    "docker system prune -a --volumes -f",
    "userdel -r deploy",
    "pacman -Rdd --noconfirm glibc",
]


@pytest.mark.parametrize("cmd", ARM4_BASES)
def test_arm4_base_is_high(cmd):
    assert classify(cmd) is HIGH, cmd


def test_rm_rf_star_slash_is_high():
    """The specific v2 gap: the old regex required whitespace/end/'/' after the target,
    so `rm -rf /*` (target '/', next char '*') fell through."""
    assert classify("rm -rf /*") is HIGH


@pytest.mark.parametrize("cmd", [
    "sudo apt update",
    "sudo apt upgrade",
    "sudo pacman -Syu",
    "sudo dnf upgrade",
    "sudo rm -rf /tmp/x",
    "sudo unknowncommand --flag",
    "doas vi /etc/passwd",
    "pkexec rm /etc/foo",
])
def test_privileged_fails_closed(cmd):
    """§2.1 — sudo and other escalation default HIGH unless on the explicit mild list."""
    assert classify(cmd) is HIGH, cmd


@pytest.mark.parametrize("cmd", [
    "sudo whoami",
    "sudo id",
    "sudo systemctl status nginx",
    "sudo systemctl is-active docker",
    "sudo journalctl -xeu service",
    "sudo mount -l",
    "sudo cat /etc/os-release",
    "sudo -v",
])
def test_explicit_mild_privileged(cmd):
    assert classify(cmd) is MILD, cmd


@pytest.mark.parametrize("cmd", [
    "ls -la",
    "git status",
    "rg foo src/",
    "cd /tmp",
    "echo hello",
    "cat /etc/passwd",          # reading the auth db is not writing it
    "getent passwd root",
    "pacman -Q glibc",
    "iptables -S -n",
    "docker system df",
])
def test_safe_commands_are_not_high(cmd):
    assert classify(cmd) is not HIGH, cmd


@pytest.mark.parametrize("cmd", [
    "opencode --remove-database",
    "docker image prune --force",
    "npm install --yes",
    "foo --delete-all",
    "foo --destroy-data",
    "foo --wipe-disk",
    "foo --erase-keys",
    "foo --purge-cache",
    "foo --drop-table",
    "foo --kill-process",
    "foo --reset-config",
    "foo --clean-build",
    "foo --auto-approve",
    "foo --sudo",
    "foo --root",
])
def test_flag_heuristic_trips(cmd):
    ok, why = flag_doubt(cmd)
    assert ok, f"{cmd!r} should trip the flag heuristic: {why}"


@pytest.mark.parametrize("cmd", [
    "ls -la",
    "git status",
    "rg foo src/",
    "cd /tmp",
    "grep --color=auto foo file",
    "tar -xf backup.tar -C /tmp --overwrite",
    "pacman -Q glibc",
])
def test_flag_heuristic_quiet(cmd):
    ok, _ = flag_doubt(cmd)
    assert not ok, cmd


def test_flag_with_equals_value():
    ok, _ = flag_doubt("foo --remove-database=true")
    assert ok


def test_short_flags_are_not_flagged():
    """Single-letter flags are too noisy to flag heuristically."""
    ok, _ = flag_doubt("rm -rf /tmp/x")
    assert not ok
