"""Tests for the shipped artefact, monitor_invariance/.

A paper whose contribution is a tool has to test the tool. Three properties
the paper rests on are checked directly, plus the package's verdicts and
arithmetic:

1. The budget-cadence identity, on random streams. `budget_invariance_test`
   implements a 1-in-r budget as `monitor(stream[::r], r*dt)` -- it ASSUMES the
   identity. So the identity is tested here against an explicit simulation
   with real timestamps (the agent acts every dt; the monitor observes every
   r-th action while the clock keeps running), not through that shortcut.
   Both clock-indexed forms are covered: exponential decay and the
   rectangular "N events in the last T seconds" window.
2. `design_report` says STARVED exactly when E < N_min, E = rate x length.
3. `sign_constraint_audit` returns BALANCED on a statistic known to fall (a
   CUSUM increment, e - k) and SIGN_CONSTRAINED on one that cannot.
"""
from __future__ import annotations

import math
import random

import pytest

from monitor_invariance import (budget_invariance_test, cadence_invariance_test,
                                sign_constraint_audit)
from monitor_invariance.design_rule import (cusum_evidence_floor, design_report,
                                            evidence_floor, max_budget,
                                            cliff_budget, starvation_budget)

LAM = math.log(2.0) / 150.0
STEP, THETA = 0.15, 0.7
# Values exact in binary, so (i + r)*dt - i*dt == r*dt exactly and the
# identity can be asserted with ==, not approximately.
DTS = (0.0, 0.5, 1.0, 1.25, 2.0, 5.0, 16.0, 64.0)


# ---- monitors, in the two forms the identity relates ------------------------
def leaky(stream, dt):
    """Wall-clock leaky integrator, as the package's callers write it."""
    s, out = 0.0, []
    for x in stream:
        s = min(1.0, s * math.exp(-LAM * dt) + STEP * x)
        out.append(s >= THETA)
    return out


def leaky_timed(events):
    """The same integrator fed (timestamp, evidence) -- what a deployed
    monitor sees when it skips actions while the clock runs."""
    s, prev, out = 0.0, None, []
    for t, x in events:
        if prev is not None:
            s *= math.exp(-LAM * (t - prev))
        s = min(1.0, s + STEP * x)
        out.append(s >= THETA)
        prev = t
    return out


def window(n, T):
    def mon(stream, dt):
        out = []
        for k in range(len(stream)):
            out.append(sum(1 for m in range(k + 1)
                           if stream[m] and (k - m) * dt < T) >= n)
        return out
    return mon


def window_timed(n, T):
    def mon(events):
        out = []
        for k, (t, _) in enumerate(events):
            out.append(sum(1 for (tm, x) in events[:k + 1] if x and t - tm < T) >= n)
        return out
    return mon


def budget_explicit(stream, dt, r, o, timed):
    """Agent acts at i*dt; the monitor observes indices o, o+r, ... with their
    REAL timestamps."""
    return timed([(i * dt, stream[i]) for i in range(o, len(stream), r)])


# ---- 1. the identity --------------------------------------------------------
@pytest.mark.parametrize("seed", range(40))
def test_budget_cadence_identity_leaky(seed):
    rng = random.Random(seed)
    stream = [1 if rng.random() < rng.uniform(0.05, 0.6) else 0
              for _ in range(rng.randint(5, 60))]
    for dt in DTS:
        for r in (1, 2, 3, 5, 10):
            for o in range(r):
                assert budget_explicit(stream, dt, r, o, leaky_timed) == \
                    leaky(stream[o::r], r * dt)


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("n,T", [(3, 10.0), (5, 600.0), (2, 64.0)])
def test_budget_cadence_identity_window(seed, n, T):
    rng = random.Random(1000 + seed)
    stream = [1 if rng.random() < 0.3 else 0 for _ in range(rng.randint(5, 40))]
    for dt in DTS:
        for r in (1, 2, 4):
            for o in range(r):
                assert budget_explicit(stream, dt, r, o, window_timed(n, T)) == \
                    window(n, T)(stream[o::r], r * dt)


def test_budget_invariance_test_agrees_with_explicit_simulation():
    """The package's shortcut gives the same fire fraction as simulating."""
    rng = random.Random(7)
    streams = [[1 if rng.random() < 0.35 else 0 for _ in range(30)] for _ in range(12)]
    cadence = 1.25
    reps = budget_invariance_test(leaky, streams, cadence, budget_grid=(1, 2, 5))
    for r, rep in reps.items():
        subs = [s[::r] for s in streams if len(s[::r]) >= 2]
        explicit = sum(any(budget_explicit(s, cadence, r, 0, leaky_timed))
                       for s in streams if len(s[::r]) >= 2) / len(subs)
        assert rep.fired_fraction[cadence] == pytest.approx(explicit)


# ---- 2. STARVED exactly when E < N_min --------------------------------------
@pytest.mark.parametrize("step,threshold", [(0.15, 0.7), (0.1, 1.0), (0.25, 0.75), (0.2, 0.5)])
def test_starved_iff_expected_evidence_below_floor(step, threshold):
    nmin = evidence_floor(threshold, step)
    for rate in (0.05, 0.1, 0.125, 0.25, 0.5):
        for length in (1, 2, 4, 7, 8, 10, 16, 20, 32, 40, 64):
            rep = design_report(step, rate, 150.0, threshold, 1.53, trace_length=length)
            starved = rep.regime.startswith("STARVED")
            assert starved == (rate * length < nmin), (step, threshold, rate, length)


def test_without_trace_length_never_starved_and_warns():
    rep = design_report(0.15, 0.3, 150.0, 0.7, 1.53)
    assert not rep.regime.startswith("STARVED")
    assert "NOT SUPPLIED" in str(rep)


# ---- 3. the sign-constraint audit -------------------------------------------
def test_cusum_increment_is_balanced():
    obs = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    rep = sign_constraint_audit(lambda e, st: [("g", e - 0.1)], obs)
    assert rep.verdict == "BALANCED"
    assert rep.n_negative == obs.count(0) and rep.n_positive == obs.count(1)


def test_count_is_sign_constrained():
    rep = sign_constraint_audit(lambda e, st: [("strikes", float(e))], [1, 0, 1, 1, 0])
    assert rep.verdict == "SIGN_CONSTRAINED"


def test_partial_and_trigger_dimension():
    ext = lambda e, st: [("up", 1.0), ("both", 1.0 if e else -1.0)]
    assert sign_constraint_audit(ext, [1, 0, 1]).verdict == "PARTIALLY_CONSTRAINED"
    # if the triggers read only the up-only dimension, the channel is useless
    assert sign_constraint_audit(ext, [1, 0, 1],
                                 trigger_dimensions=["up"]).verdict == "SIGN_CONSTRAINED"


def test_no_signals():
    assert sign_constraint_audit(lambda e, st: [], [1, 0]).verdict == "NO_SIGNALS"


# ---- cadence verdicts --------------------------------------------------------
def test_leaky_integrator_is_a_cliff_and_cusum_is_invariant():
    streams = [[1 if (i % 3 == 0) else 0 for i in range(40)] for _ in range(6)]

    def cusum(stream, dt):
        g, out = 0.0, []
        for x in stream:
            g = max(0.0, g + x - 0.1)
            out.append(g >= 1.0)
        return out

    assert cadence_invariance_test(leaky, streams).verdict == "CLIFF"
    assert cadence_invariance_test(cusum, streams).verdict == "INVARIANT"


def test_monitor_must_return_one_decision_per_observation():
    with pytest.raises(ValueError):
        cadence_invariance_test(lambda s, dt: [False], [[1, 0, 1]])


# ---- arithmetic --------------------------------------------------------------
def test_evidence_floors():
    assert evidence_floor(0.7, 0.15) == 5
    assert evidence_floor(0.6, 0.15) == 4          # 0.6/0.15 is 3.999... in floats
    assert cusum_evidence_floor(1.0, 0.1) == 2
    assert starvation_budget(10, 5) == 2.0


def test_max_budget_is_the_binding_bound():
    rc = cliff_budget(0.15, 0.3, 150.0, 0.7, 1.53)
    for length in (5, 20, 200):
        rs = starvation_budget(0.3 * length, evidence_floor(0.7, 0.15))
        assert max_budget(0.15, 0.3, 150.0, 0.7, 1.53, length) == pytest.approx(min(rc, rs))
    assert max_budget(0.15, 0.3, 150.0, 0.7, 1.53) == pytest.approx(rc)
