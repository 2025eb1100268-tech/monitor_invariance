"""The design rule: where is my cliff, and how much can I afford to skip?

For a leaky integrator

    s <- s * exp(-lambda * delta) + a * e     lambda = ln2 / H

with step `a`, half-life `H` seconds, observation interval `delta` seconds and
input events arriving with probability `p` per observation, the state settles
around

    E[s_inf] = a * p / (1 - exp(-lambda * delta))

Setting that equal to the fire threshold `theta` and solving for delta gives the
critical observation interval -- the point where the monitor stops being able to
hold itself above threshold:

    delta* = -(H / ln2) * ln(1 - a * p / theta)

Read it as a design rule. The monitor has a usable regime only if its expected
observation interval sits near delta*. Sit well below and it latches on
permanently; well above and it never fires. And because a cost-constrained
deployment that samples 1-in-r actions multiplies the interval by r, the same
formula bounds the monitoring budget you can afford:

    r_cliff = delta* / cadence

If `a * p >= theta` the integrator saturates regardless of interval: delta* is
infinite and the monitor is a latch, not a detector. That is reported, not
hidden.

THE SECOND BOUND, AND USUALLY THE BINDING ONE. The cliff bound assumes the
trace is long enough to reach equilibrium. Agent traces are not. Any
accumulator needs a minimum number of evidence events before it can reach
threshold at all -- for a leaky integrator

    N_min = ceil(theta / a)                    (5 for a=0.15, theta=0.7)

and a trace carrying E evidence events leaves about E / r of them under a
1-in-r budget, so

    r_starv = E / N_min

This bound depends on no clock and no calibration, so it applies to sample-time
monitors too, and it is independent of the band. On measured agent debugging
traces (Phase 9C: 4-20 actions, 0-10 errors) it binds on every run and its
median is below 1 -- the median trace cannot drive the accumulator to threshold
even with nothing skipped. Report min(r_cliff, r_starv), never r_cliff alone.
An earlier version of this module reported r_cliff as "max safe budget" and so
overstated the headroom by roughly a factor of five on the paper's own monitor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def equilibrium_level(step: float, rate: float, half_life: float,
                      interval: float) -> float:
    """E[s_inf] for the leaky integrator at a given observation interval."""
    if interval <= 0:
        return math.inf if rate > 0 else 0.0
    lam = math.log(2.0) / half_life
    d = math.exp(-lam * interval)
    return step * rate / (1.0 - d) if d < 1.0 else math.inf


def critical_interval(step: float, rate: float, half_life: float,
                      threshold: float) -> Optional[float]:
    """delta* in seconds, or None if the integrator saturates at any interval.

    None means `step * rate >= threshold`: even with the decay running flat out
    between observations the input outruns it. Such a monitor has no
    moment-detection regime at all -- it is a latch.
    """
    x = step * rate / threshold
    if x >= 1.0:
        return None
    return -(half_life / math.log(2.0)) * math.log(1.0 - x)


def cliff_budget(step: float, rate: float, half_life: float, threshold: float,
                 cadence: float) -> Optional[float]:
    """Largest 1-in-r rate before the monitor falls off its cliff (r_cliff).

    Assumes the trace is long enough to reach equilibrium. On agent traces it
    usually is not, and `starvation_budget` binds first -- see `max_budget`.
    """
    d = critical_interval(step, rate, half_life, threshold)
    if d is None:
        return None
    return d / cadence if cadence > 0 else math.inf


# Kept under its old name so existing callers keep working. It now means what
# it always should have: the binding bound when trace length is supplied.
def safe_budget(step: float, rate: float, half_life: float, threshold: float,
                cadence: float, trace_length: Optional[int] = None) -> Optional[float]:
    return max_budget(step, rate, half_life, threshold, cadence, trace_length)


def evidence_floor(threshold: float, step: float) -> int:
    """N_min: evidence events needed before a leaky integrator can reach
    threshold at all, with no decay (the best case). ceil(theta / a)."""
    return math.ceil(threshold / step - 1e-12)


def cusum_evidence_floor(h: float, k: float) -> int:
    """N_min for a sample-time CUSUM g <- max(0, g + e - k), fire at g >= h,
    with evidence e in {0, 1}: ceil(h / (1 - k))."""
    return math.ceil(h / (1.0 - k) - 1e-12)


def starvation_budget(expected_evidence: float, n_min: int) -> float:
    """r_starv = E / N_min. Below 1 means the monitor cannot fire on a trace
    of this size even when it sees every action."""
    return expected_evidence / n_min


def max_budget(step: float, rate: float, half_life: float, threshold: float,
               cadence: float, trace_length: Optional[int] = None) -> Optional[float]:
    """min(r_cliff, r_starv). Supply trace_length; without it only the cliff
    bound is computable and the result overstates the headroom."""
    rc = cliff_budget(step, rate, half_life, threshold, cadence)
    if trace_length is None:
        return rc
    rs = starvation_budget(rate * trace_length, evidence_floor(threshold, step))
    return rs if rc is None else min(rc, rs)


@dataclass
class DesignReport:
    step: float
    rate: float
    half_life: float
    threshold: float
    cadence: float
    critical_interval_s: Optional[float]
    safe_budget_r: Optional[float]
    level_at_cadence: float
    regime: str
    trace_length: Optional[int] = None
    cliff_budget_r: Optional[float] = None
    starvation_budget_r: Optional[float] = None
    n_min: Optional[int] = None

    def __str__(self) -> str:
        ci = ("saturates at every interval"
              if self.critical_interval_s is None
              else f"{self.critical_interval_s:.1f} s")
        cb = ("n/a -- latch" if self.cliff_budget_r is None
              else f"1-in-{self.cliff_budget_r:.1f}")
        lines = [
            "leaky-integrator design rule",
            f"  step a           : {self.step}",
            f"  input rate p     : {self.rate}",
            f"  half-life H      : {self.half_life} s",
            f"  threshold theta  : {self.threshold}",
            f"  agent cadence    : {self.cadence} s between actions",
            f"  trace length     : "
            + (f"{self.trace_length} actions" if self.trace_length else
               "NOT SUPPLIED -- starvation bound cannot be computed"),
            "",
            f"  critical interval: {ci}",
            f"  equilibrium level at cadence: {self.level_at_cadence:.2f} "
            f"unclamped, vs threshold {self.threshold} "
            f"({self.level_at_cadence / self.threshold:.0f}x over)",
            "",
            f"  cliff bound      : {cb}",
        ]
        if self.starvation_budget_r is not None:
            lines.append(
                f"  starvation bound : 1-in-{self.starvation_budget_r:.1f}  "
                f"(N_min = {self.n_min} evidence events; "
                f"~{self.rate * self.trace_length:.1f} expected per trace)")
        binding = ("starvation" if (self.starvation_budget_r is not None and
                   (self.cliff_budget_r is None or
                    self.starvation_budget_r <= self.cliff_budget_r)) else "cliff")
        if self.safe_budget_r is None:
            lines.append("  MAX SAFE BUDGET  : n/a -- monitor is a latch, not a detector")
        elif self.safe_budget_r < 1.0:
            lines.append(f"  MAX SAFE BUDGET  : BELOW 1 ({self.safe_budget_r:.2f}) -- "
                         f"the monitor cannot reach threshold on a trace this "
                         f"short even with nothing skipped. Binding: {binding}.")
        else:
            lines.append(f"  MAX SAFE BUDGET  : 1-in-{self.safe_budget_r:.1f} "
                         f"(binding: {binding})")
        if self.trace_length is None:
            lines.append("  WARNING          : without trace_length this is the cliff "
                         "bound only, which overstates the headroom on short traces.")
        lines += ["", f"  regime: {self.regime}"]
        return "\n".join(lines)


def design_report(step: float, rate: float, half_life: float,
                  threshold: float, cadence: float,
                  trace_length: Optional[int] = None) -> DesignReport:
    d = critical_interval(step, rate, half_life, threshold)
    lvl = equilibrium_level(step, rate, half_life, cadence)
    nmin = evidence_floor(threshold, step)
    rc = cliff_budget(step, rate, half_life, threshold, cadence)
    rs = (starvation_budget(rate * trace_length, nmin)
          if trace_length is not None else None)
    if d is None:
        regime = ("SATURATED -- step * rate exceeds the threshold, so the state "
                  "reaches the clamp at any observation interval and stays "
                  "there. Lower the step, raise the threshold, or move to a "
                  "sample-time statistic.")
    elif cadence < d / 3.0:
        regime = (f"LATCHED -- the agent acts {d/cadence:.0f}x faster than the "
                  f"critical interval, so the monitor sits pinned above "
                  f"threshold and carries almost no information. Sampling "
                  f"1-in-{d/cadence:.0f} would put it at the boundary, which is "
                  f"the worst place to be.")
    elif cadence > d * 3.0:
        regime = ("DEAD -- observations arrive too far apart for the state to "
                  "accumulate; it decays away between events and the threshold "
                  "is never reached.")
    else:
        regime = ("BOUNDARY -- cadence is within 3x of the critical interval. "
                  "Firing behaviour here is highly sensitive to latency "
                  "variation and to any change in monitoring budget.")
    # The regimes above describe a trace long enough to reach equilibrium. If
    # the trace cannot even supply N_min evidence events, that description is
    # false for it, and saying "LATCHED" next to "cannot reach threshold" would
    # be a contradiction in the tool's own output.
    if rs is not None and rs < 1.0:
        regime = (f"STARVED -- a {trace_length}-action trace carries about "
                  f"{rate * trace_length:.1f} evidence events, fewer than the "
                  f"{nmin} this monitor needs to reach threshold, so it cannot "
                  f"fire on this trace even when it sees every action. (On a "
                  f"trace long enough to reach equilibrium it would instead be: "
                  f"{regime.split(' --')[0]}.) Lower theta/a, or accept that "
                  f"the monitor is silent on short episodes.")
    return DesignReport(step, rate, half_life, threshold, cadence, d,
                        max_budget(step, rate, half_life, threshold, cadence,
                                   trace_length),
                        lvl, regime, trace_length, rc, rs, nmin)
