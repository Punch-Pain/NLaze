"""Front-gate danger classifier — adversarial coverage from arm 4."""
import pytest
from nlaze.danger import classify, HIGH, MILD, NONE

@pytest.mark.parametrize("cmd", [
    "rm -rf / --no-preserve-root",
    "rm -rf /home/abdo",
    "rm -rf ~",
    "rm -r /usr",
    "dd if=/dev/zero of=/dev/sda",
    "dd if=/dev/zero of=/dev/nvme0n1p2",
    ":(){ :|:& };:",
    "mkfs.ext4 /dev/sda2",
    "wipefs -a /dev/nvme0n1p2",
    "chmod -R 777 /",
    "git push origin main --force",
    "git push --force-with-lease",
    "git reset --hard HEAD",
    "crontab -r",
    "kill -9 -1",
    "hdparm --security-erase ALL /dev/sda",
    "dd if=img.iso of=/dev/mmcblk0",
    "forkbomb",
])
def test_high_patterns(cmd):
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
    "ls -la",
    "cd ..",
    "grep -r foo .",
    "rm foo.txt",                       # single file delete, no --no-preserve-root, no system root
])
def test_non_high_patterns(cmd):
    assert classify(cmd) is not HIGH, cmd

@pytest.mark.parametrize("cmd, sev", [
    ("sudo whoami", MILD),
    ("sudo systemctl status nginx", MILD),
    ("ls -la", NONE),
    ("git push origin main", NONE),
])
def test_severity_exact(cmd, sev):
    assert classify(cmd) is sev, cmd

def test_mixed_with_dangerous_suffix_is_high():
    assert classify("sudo whoami; rm -rf /") is HIGH

@pytest.mark.parametrize("cmd", [
    "git push origin main",  # normal push
])
def test_normal_push_not_high(cmd):
    assert classify(cmd) is not HIGH


def test_package_upgrade_is_high():
    """§2.1 — a package upgrade is a system change, not a read-only query.
    The v2 table had it as MILD; on an Arch machine that was the invisible danger."""
    assert classify("sudo pacman -Syu") is HIGH
    assert classify("sudo apt upgrade") is HIGH
