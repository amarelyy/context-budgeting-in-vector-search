"""Phase 3 (3.2): unit test for signals.py with the worked example from CONTEXT.md.

Distances 0.20, 0.21, 0.90 -> mean 0.4367, margin_mc = 0.01, var_local = 0.1074.
Plain asserts, no pytest needed. Run from the repo root:
    .\.venv\Scripts\python.exe tests\test_signals.py
"""
import math
import sys

sys.path.insert(0, "src")
from signals import compute_signals  # noqa: E402


def close(a, b, tol=1e-4):
    return abs(a - b) < tol


def test_worked_example():
    s = compute_signals([0.20, 0.21, 0.90])
    assert close(s["margin_mc"], 0.01), s["margin_mc"]
    assert close(s["var_local"], 0.1074), s["var_local"]
    assert close(s["top1_dist"], 0.20), s["top1_dist"]
    assert s["elbow_idx"] == 2, s["elbow_idx"]          # big jump after rank 2
    assert 0.0 <= s["entropy"] <= 1.0, s["entropy"]


def test_unsorted_input_gives_same_result():
    assert compute_signals([0.90, 0.20, 0.21]) == compute_signals([0.20, 0.21, 0.90])


def test_entropy_extremes():
    flat = compute_signals([0.5, 0.5, 0.5, 0.5])
    assert close(flat["entropy"], 1.0), flat["entropy"]      # all equal -> max entropy
    peak = compute_signals([0.1, 1.9, 1.9, 1.9])
    assert peak["entropy"] < 0.01, peak["entropy"]           # one clear winner -> ~0


if __name__ == "__main__":
    tests = [test_worked_example, test_unsorted_input_gives_same_result, test_entropy_extremes]
    for t in tests:
        t()
        print("PASS", t.__name__)
    s = compute_signals([0.20, 0.21, 0.90])
    print("worked example:", {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})
    print("mean check:", round(sum([0.20, 0.21, 0.90]) / 3, 4), "| expected 0.4367")
