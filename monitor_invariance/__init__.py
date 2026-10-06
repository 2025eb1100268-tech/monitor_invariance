"""monitor_invariance -- is your runtime monitor calibrated to the clock or to
its own observations?

A runtime monitor that carries internal state has to decide what its state
decays against. Two choices, and only two:

  WALL-CLOCK    state *= exp(-lambda * dt_seconds)   -- a half-life in seconds
  SAMPLE-TIME   state updated once per observation   -- a CUSUM, a counter

The choice usually gets made once, early, without anyone writing it down. It
determines whether the monitor has a usable operating regime at the rate the
agent actually acts. A wall-clock-calibrated leaky integrator sampled at agent
cadence is bistable: below a critical inter-observation interval it latches into
a permanent alarm, above it the state never reaches threshold at all, and the
band in between is narrow. A sample-time integrator over the identical stream is
exactly invariant to that interval.

This package gives a monitor author three things they can run against their own
monitor, without adopting anything else from the paper:

  1. cadence_invariance_test  -- sweep the inter-observation interval and report
     INVARIANT (sample-time calibrated) or CLIFF (wall-clock calibrated).
  2. sign_constraint_audit    -- can your evidence extractor ever emit
     exculpatory evidence, or is every signal incriminating? A monitor whose
     evidence channel is sign-constrained cannot de-escalate no matter which
     integrator sits behind it.
  3. critical_interval / safe_budget -- the closed-form design rule: given input
     rate, step size, half-life and threshold, where is your cliff, and how much
     of the action stream can you afford to skip before you fall off it?

Reference: Modgil, "Bistable by Construction: Wall-Clock-Calibrated State
Monitors Have No Moment-Detection Regime at Agent Cadence", arXiv:2606.19386.

Apache-2.0.
"""
from .invariance import (
    CadenceReport,
    cadence_invariance_test,
    budget_invariance_test,
)
from .sign_check import SignReport, sign_constraint_audit
from .design_rule import (
    critical_interval,
    equilibrium_level,
    safe_budget,
    cliff_budget,
    max_budget,
    evidence_floor,
    cusum_evidence_floor,
    starvation_budget,
    design_report,
)

__version__ = "0.1.0"

__all__ = [
    "CadenceReport",
    "cadence_invariance_test",
    "budget_invariance_test",
    "SignReport",
    "sign_constraint_audit",
    "critical_interval",
    "equilibrium_level",
    "safe_budget",
    "cliff_budget",
    "max_budget",
    "evidence_floor",
    "cusum_evidence_floor",
    "starvation_budget",
    "design_report",
]
