"""End-to-end fish -> daemon -> correction -> execution test inside an isolated
bubblewrap (bwrap) container sandbox with a read-only host root, tmpfs /tmp /home /run,
and no host /home or USB mounts (spec §10, plan Phase 0.4 & Phase 3.2)."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
NLAZE_PKG = ROOT / "nlaze"

HAS_BWRAP = shutil.which("bwrap") is not None
HAS_FISH = shutil.which("fish") is not None


@pytest.mark.skipif(not (HAS_BWRAP and HAS_FISH), reason="bwrap and fish required for sandbox e2e")
def test_fish_daemon_e2e_in_isolated_sandbox():
    fish_script = r"""
# 0. Verify sandbox isolation: no host /home/abdo or /run/media mounted, /tmp/nlaze is read-only
if test -e /home/abdo -o -e /run/media
    echo "FAIL: host paths visible inside sandbox"
    exit 1
end
if touch /tmp/nlaze/evil 2>/dev/null
    echo "FAIL: /tmp/nlaze is writable"
    exit 1
end

# Set up wrapper for nlaze CLI using in-sandbox python3
mkdir -p /home/sandbox/bin
printf '#!/bin/sh\nexec /usr/bin/python3 -m nlaze.cli "$@"\n' > /home/sandbox/bin/nlaze
chmod +x /home/sandbox/bin/nlaze
set -gx PATH /home/sandbox/bin $PATH
set -gx NLAZE_BIN /home/sandbox/bin/nlaze
set -gx NLAZE_SOCKET_PATH /tmp/nlaze-run/daemon.sock
set -gx NLAZE_USAGE_PATH /home/sandbox/usage.sqlite
set -gx NLAZE_POLICY_PATH /home/sandbox/policy.sqlite

# Initialize a scratch git repo in /tmp/work (tmpfs)
mkdir -p /tmp/work
cd /tmp/work
git init -q /tmp/work
git -C /tmp/work config user.email "sandbox@local"
git -C /tmp/work config user.name "Sandbox"
echo "hello" > /tmp/work/README.md
git -C /tmp/work add README.md
git -C /tmp/work commit -q -m "init"

# 1. Start the nlaze daemon in background inside the sandbox
nlaze daemon --socket $NLAZE_SOCKET_PATH &
set -l daemon_pid $last_pid
for i in (seq 1 30)
    if test -S $NLAZE_SOCKET_PATH
        break
    end
    sleep 0.05
end
if not test -S $NLAZE_SOCKET_PATH
    echo "FAIL: daemon socket did not appear"
    exit 1
end

# Load the fish hook
source /tmp/nlaze/fish/nlaze.fish

# 2. Valid command with nonzero exit (grep no match) -> passthrough (no output)
false
set -l out_valid ( begin; false; __nlaze_postexec "grep missing_xyz README.md"; end )
if test -n "$out_valid"
    echo "FAIL: valid failing command should passthrough, got: $out_valid"
    exit 1
end

# 3. Typo (`git statts`) -> confirm prompt -> user types 'y' -> runs `git status`
set -l out_confirm ( printf 'y\n' | begin; false; __nlaze_postexec "git statts"; end )
if not string match -q "*On branch*" "$out_confirm"
    echo "FAIL: expected git status output on confirm accept, got: $out_confirm"
    exit 1
end

# 4. Unallowlisted head (`pythno3 -c "print(12345)"`) -> certify_head prompt -> user types 'yes'
set -l out_cert_head ( printf 'yes\n' | begin; false; __nlaze_postexec "pythno3 -c 'print(12345)'"; end )
if not string match -q "*one-time request*12345*" "$out_cert_head"
    echo "FAIL: expected head certification + execution, got: $out_cert_head"
    exit 1
end
set -l al_list (nlaze allowlist list)
if not string match -q "*python3*" "$al_list"
    echo "FAIL: python3 not added to allowlist: $al_list"
    exit 1
end

# 5. Tier B destructive flag (`pythno3 --remove-cache`) -> certify_flag prompt -> user types 'no' -> deny list
set -l out_cert_flag ( printf 'no\n' | begin; false; __nlaze_postexec "pythno3 --remove-cache"; end )
set -l dn_list (nlaze danger list)
if not string match -q "*python3 --remove-cache*" "$dn_list"
    echo "FAIL: expected python3 --remove-cache in deny list, got: $dn_list"
    exit 1
end

# 6. High-danger candidate (`git pusch --force origin main`) -> double confirm rejects single 'y'
set -l out_danger ( printf 'y\n' | begin; false; __nlaze_postexec "git pusch --force origin main"; end )
if not string match -q "*HIGH DANGER*" "$out_danger"
    echo "FAIL: expected HIGH DANGER warning panel, got: $out_danger"
    exit 1
end

# 7. Direct-serve Auto & Near-neighbor Strike demotion
python3 -c '
from nlaze.usage import Store
s = Store("/home/sandbox/usage.sqlite")
for _ in range(20):
    s.record_accept("gti status", "git status")
for _ in range(110):
    s.tick()
'
set -l out_auto ( begin; false; __nlaze_postexec "gti status"; end | string join " " )
if not string match -q "*nlaze auto*git status*On branch*" "$out_auto"
    echo "FAIL: expected direct-serve auto execution, got: $out_auto"
    exit 1
end

# User now runs an alternative from the offer (`git stash`) with exit 0 -> 1 strike!
true
__nlaze_postexec "git stash"

# Next `gti status` must be demoted from auto to confirm!
set -l out_demoted ( printf 'n\n' | begin; false; __nlaze_postexec "gti status"; end | string join " " )
if string match -q "*nlaze auto*" "$out_demoted"
    echo "FAIL: expected strike demotion from auto to confirm, got: $out_demoted"
    exit 1
end

# 8. Kill switch (`NLAZE_DISABLE=1`) disables hook immediately
set -g NLAZE_DISABLE 1
set -l out_disabled ( begin; false; __nlaze_postexec "git statts"; end )
if test -n "$out_disabled"
    echo "FAIL: kill switch did not silence hook: $out_disabled"
    exit 1
end

kill $daemon_pid 2>/dev/null
echo "SANDBOX_E2E_OK"
"""
    cmd = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--ro-bind", "/", "/",
        "--tmpfs", "/tmp",
        "--tmpfs", "/home",
        "--tmpfs", "/run",
        "--dev", "/dev",
        "--proc", "/proc",
        "--ro-bind", str(NLAZE_PKG), "/tmp/nlaze",
        "--setenv", "HOME", "/home/sandbox",
        "--setenv", "PYTHONPATH", "/tmp",
        "--chdir", "/tmp",
        "/usr/bin/fish", "-c", fish_script,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SANDBOX_E2E_OK" in proc.stdout
