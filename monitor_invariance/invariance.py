"""The cadence-invariance test.

You supply a monitor as a callable. We sweep the inter-observation interval and
tell you whether its firing behaviour is invariant (sample-time calibrated) or
falls off a cliff (wall-clock calibrated).

    def my_monitor(stream, dt):
        '''stream: the observations, in order. dt: seconds between successive
        observations. Return one truthy/falsy value per observation: did the
        monitor consider itself firing at that observation?'''
        ...

    report = cadence_invariance_test(my_monitor, streams)
    print(report)                 # human-readable
    report.verdict                # 'INVARIANT' | 'CLIFF' | 'MIXED' | 'NEVER_FIRES'

Nothing here imports the HEART engine or anything else from the paper; it is
stdlib-only so it can be dropped into any codebase.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

# The dt grid from the paper, in seconds. Spans four orders of magnitude so the
# transition is bracketed wherever it sits.
DEFAULT_DT_GRID: Tuple[float, ...] = (0.0, 1.0, 5.0, 15.0, 30.0, 60.0,
                                      150.0, 300.0, 600.0)
DEFAULT_BUDGET_GRID: Tuple[int, ...] = (1, 2, 5, 10, 20, 50)

# A monitor has "left the alarm regime" when fewer than half the observations
# after its first fire are still firing. Same rule as the paper's Phase 5/8.
PERSISTENCE_DROP = 50.0

MonitorFn = Callable[[Sequence[Any], float], Sequence[Any]]


def _persistence(fires: Sequence[Any]) -> Tuple[bool, float, int]:
    """(fired at all, post-first-fire persistence %, fire count)."""
    flags = [bool(f) for f in fires]
    first = next((i for i, f in enumerate(flags) if f), None)
    if first is None:
        return False, 0.0, 0
    post = flags[first:]
    return True, 100.0 * sum(post) / len(post), sum(flags)


@dataclass
class CadenceReport:
    verdict: str
    dt_grid: Tuple[float, ...]
    persistence: Dict[float, float]
    fire_counts: Dict[float, float]
    fired_fraction: Dict[float, float]
    critical_interval: Tuple[float, float]
    n_streams: int
    saturated: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_wall_clock(self) -> bool:
        return self.verdict == "CLIFF"

    def __str__(self) -> str:
        lo, hi = self.critical_interval
        band = "none found" if hi == math.inf else (
            f"({lo:g}, {hi:g}] s" if lo else f"(0, {hi:g}] s")
        lines = [
            "cadence-invariance test",
            f"  streams          : {self.n_streams}",
            f"  verdict          : {self.verdict}",
            f"  critical interval: {band}",
            "",
            "  dt (s)   fired    persistence %   median fires",
            "  " + "-" * 46,
        ]
        for dt in self.dt_grid:
            lines.append(
                f"  {dt:7g}  {self.fired_fraction[dt]*100:5.0f}%  "
                f"{self.persistence[dt]:13.1f}   {self.fire_counts[dt]:12.1f}")
        lines += ["", "  " + _EXPLAIN[self.verdict]]
        if self.saturated:
            lines += ["", "  " + _SATURATION_WARNING]
        return "\n".join(lines)


_EXPLAIN = {
    "INVARIANT": ("Firing behaviour does not change with the observation "
                  "interval. The monitor is SAMPLE-TIME calibrated: its state "
                  "advances per observation and never reads a clock. Cadence "
                  "changes and monitoring-budget changes cost it sensitivity, "
                  "not a regime."),
    "CLIFF": ("Firing behaviour collapses across a narrow band of observation "
              "intervals. The monitor is WALL-CLOCK calibrated. Below the band "
              "it latches on; above it, it never fires. Anything that changes "
              "the spacing of its observations -- a faster agent, a slower "
              "one, or a decision to sample 1-in-r actions to save money -- "
              "moves it across that boundary without anyone editing a "
              "threshold."),
    "MIXED": ("Behaviour changes with dt but not as a clean two-regime "
              "collapse. There is clock-dependence in here; inspect the table "
              "and the per-stream detail before trusting the monitor at any "
              "particular cadence."),
    "NEVER_FIRES": ("The monitor never fired at any dt on these streams. The "
                    "test is uninformative -- supply streams that contain the "
                    "signal the monitor is meant to catch."),
}

_SATURATION_WARNING = (
    "SEPARATE WARNING -- SATURATED: the monitor is still firing on nearly "
    "every stream at the largest interval tested, so on these streams it is "
    "pinned on and carries little information. This is orthogonal to the "
    "verdict above: calibration is about whether behaviour MOVES with the "
    "interval, saturation is about the LEVEL it sits at. A sample-time monitor "
    "can be perfectly invariant and still useless because its threshold is too "
    "low; a wall-clock monitor can have a cliff and a sensible level. Fix them "
    "separately -- retuning a threshold will not remove a cliff, and "
    "recalibrating to observation time will not fix a threshold.")


def cadence_invariance_test(
    monitor: MonitorFn,
    streams: Sequence[Sequence[Any]],
    dt_grid: Sequence[float] = DEFAULT_DT_GRID,
    persistence_drop: float = PERSISTENCE_DROP,
) -> CadenceReport:
    """Sweep the inter-observation interval and classify the monitor.

    `monitor(stream, dt) -> one truthy value per observation`.
    `streams` should be several real traces, not one.
    """
    if not streams:
        raise ValueError("supply at least one stream")
    dt_grid = tuple(float(d) for d in dt_grid)

    persist: Dict[float, float] = {}
    fires: Dict[float, float] = {}
    fired_frac: Dict[float, float] = {}
    per_stream: Dict[float, List[Tuple[bool, float, int]]] = {}

    for dt in dt_grid:
        rows = []
        for s in streams:
            out = monitor(s, dt)
            if len(out) != len(s):
                raise ValueError(
                    f"monitor returned {len(out)} decisions for a stream of "
                    f"{len(s)} observations; return one per observation")
            rows.append(_persistence(out))
        per_stream[dt] = rows
        crossers = [r for r in rows if r[0]]
        persist[dt] = statistics.median([r[1] for r in crossers]) if crossers else 0.0
        fires[dt] = statistics.median([r[2] for r in rows])
        fired_frac[dt] = len(crossers) / len(rows)

    # critical interval: first grid dt whose median persistence drops below the
    # bar. Reported as (previous grid point, that dt].
    crit: Tuple[float, float] = (dt_grid[-1], math.inf)
    for i, dt in enumerate(dt_grid):
        if persist[dt] < persistence_drop:
            crit = (dt_grid[i - 1] if i else 0.0, dt)
            break

    lo_dt, hi_dt = dt_grid[0], dt_grid[-1]
    never = all(fired_frac[d] == 0.0 for d in dt_grid)
    fires_lo = fired_frac[lo_dt] >= 0.5 and persist[lo_dt] >= persistence_drop

    # Calibration and saturation are ORTHOGONAL and must not be conflated:
    # the verdict is about whether behaviour MOVES with the interval, the
    # saturation flag is about the LEVEL it sits at. A sample-time CUSUM with a
    # low threshold is invariant AND saturated; reporting only "always fires"
    # would hide the very property the test exists to measure.
    saturated = fired_frac[hi_dt] >= 0.9 and persist[hi_dt] >= 90.0

    flat = (_flat(persist, dt_grid) and _flat(fires, dt_grid)
            and _flat(fired_frac, dt_grid))

    if never:
        verdict = "NEVER_FIRES"
    elif flat:
        verdict = "INVARIANT"
    elif fires_lo and crit[1] != math.inf and fired_frac[hi_dt] < 0.5:
        verdict = "CLIFF"
    else:
        verdict = "MIXED"

    return CadenceReport(
        verdict=verdict, dt_grid=dt_grid, persistence=persist,
        fire_counts=fires, fired_fraction=fired_frac, critical_interval=crit,
        n_streams=len(streams), saturated=saturated,
        detail={"per_stream": per_stream, "persistence_drop": persistence_drop},
    )


def _flat(d: Dict[float, float], grid: Sequence[float], tol: float = 1e-9) -> bool:
    vals = [d[g] for g in grid]
    return max(vals) - min(vals) <= tol


def budget_invariance_test(
    monitor: MonitorFn,
    streams: Sequence[Sequence[Any]],
    cadence: float,
    budget_grid: Sequence[int] = DEFAULT_BUDGET_GRID,
    persistence_drop: float = PERSISTENCE_DROP,
) -> Dict[int, CadenceReport]:
    """The cost-constrained version, and the one that bites in practice.

    Monitoring every action of every agent is expensive, so deployments sample
    1-in-r. For a wall-clock-calibrated monitor that multiplies the elapsed time
    between observations by r -- the budget knob and the cadence knob are the
    same knob. This runs the invariance test at each budget, holding the agent's
    real `cadence` (seconds between actions) fixed.

    Returns {r: CadenceReport}, where each report's dt axis is the *nominal*
    per-action interval; the monitor sees r times that.
    """
    out: Dict[int, CadenceReport] = {}
    for r in budget_grid:
        subs = [s[::r] for s in streams]
        subs = [s for s in subs if len(s) >= 2]
        if not subs:
            continue

        def sampled(stream, dt, _r=r):
            return monitor(stream, dt * _r)

        out[r] = cadence_invariance_test(
            sampled, subs, dt_grid=(cadence,), persistence_drop=persistence_drop)
    return out
