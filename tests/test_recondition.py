"""Near-neighbor reconditioning (spec §5.2).

The Auto path has no explicit reject, so the user's next command is the only feedback
channel. A strike = the user runs an alternative from a recent offer for the same input.
"""
import pytest

from nlaze.usage import Store
from nlaze.recondition import Reconditioner


@pytest.fixture
def rc(tmp_path):
    return Reconditioner(Store(tmp_path / "u.sqlite"), window=10, max_offers=64)


def test_no_strike_when_user_runs_the_autocorrection(rc):
    rc.record_offer("gti status", "git status", ["git status", "git stash"])
    assert rc.observe("git status") == []


def test_strike_when_user_runs_an_alternative(rc):
    rc.record_offer("gti status", "git status", ["git status", "git stash"])
    strikes = rc.observe("git stash")
    assert len(strikes) == 1
    _oh, corr = strikes[0]
    assert corr == "git status"


def test_no_strike_for_unrelated_command(rc):
    rc.record_offer("gti status", "git status", ["git status", "git stash"])
    assert rc.observe("ls -la") == []


def test_strike_demotes_auto(rc):
    rc.record_offer("gti status", "git status", ["git status", "git stash"])
    strikes = rc.observe("git stash")
    rc.apply(strikes)
    oh, corr = strikes[0]
    assert not rc.auto_allowed(oh, corr)


def test_second_strike_resets_pair(rc):
    rc.record_offer("gti status", "git status", ["git status", "git stash"])
    rc.apply(rc.observe("git stash"))
    rc.record_offer("gti status", "git status", ["git status", "git stash"])
    rc.apply(rc.observe("git stash"))
    oh = rc.store.hash_input("gti status")
    assert not rc.auto_allowed(oh, "git status")
    assert rc.strike_count(oh, "git status") >= 2


def test_expired_offer_does_not_strike(rc):
    rc.record_offer("gti status", "git status", ["git status", "git stash"])
    for _ in range(11):        # window is 10 commands
        rc.store.tick()
    assert rc.observe("git stash") == []


def test_ring_buffer_is_bounded(tmp_path):
    rc = Reconditioner(Store(tmp_path / "u.sqlite"), window=10, max_offers=4)
    for i in range(10):
        rc.record_offer(f"in{i}", f"cmd{i}", [f"cmd{i}", "other"])
    assert rc.offer_count() == 4
