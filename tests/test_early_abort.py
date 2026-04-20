"""Tests for FailureTracker (early abort on consecutive failures)."""

from __future__ import annotations

from agentic_swarm_bench.scenarios.player import FailureTracker


class TestFailureTracker:
    def test_no_threshold_never_aborts(self):
        tracker = FailureTracker(threshold=None)
        for _ in range(100):
            tracker.record_failure(0, "error")
        assert not tracker.aborted

    def test_threshold_triggers_abort(self):
        tracker = FailureTracker(threshold=3)
        tracker.record_failure(0, "err1")
        assert not tracker.aborted
        tracker.record_failure(0, "err2")
        assert not tracker.aborted
        tracker.record_failure(0, "err3")
        assert tracker.aborted

    def test_success_resets_counter(self):
        tracker = FailureTracker(threshold=3)
        tracker.record_failure(0, "err1")
        tracker.record_failure(0, "err2")
        tracker.record_success(0)
        tracker.record_failure(0, "err3")
        tracker.record_failure(0, "err4")
        assert not tracker.aborted

    def test_per_slot_independence(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure(0, "slot0-err1")
        tracker.record_failure(1, "slot1-err1")
        assert not tracker.aborted

        tracker.record_failure(1, "slot1-err2")
        assert tracker.aborted

    def test_abort_message_includes_details(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_failure(1, "connection refused")
        tracker.record_failure(1, "timeout")
        msg = tracker.abort_message()
        assert "slot 1" in msg
        assert "2" in msg

    def test_record_failure_returns_true_on_abort(self):
        tracker = FailureTracker(threshold=1)
        triggered = tracker.record_failure(0, "err")
        assert triggered is True
        assert tracker.aborted

    def test_record_failure_returns_false_before_threshold(self):
        tracker = FailureTracker(threshold=3)
        assert tracker.record_failure(0, "err") is False
        assert tracker.record_failure(0, "err") is False

    def test_aborted_property_when_fresh(self):
        tracker = FailureTracker(threshold=5)
        assert not tracker.aborted

    def test_success_on_fresh_slot(self):
        tracker = FailureTracker(threshold=2)
        tracker.record_success(0)
        tracker.record_failure(0, "err")
        assert not tracker.aborted
