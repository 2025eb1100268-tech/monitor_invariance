"""The sign-constraint audit.

A separate failure mode from calibration, and one that no amount of retuning
fixes. Ask of your monitor's evidence extractor: *can it ever emit exculpatory
evidence?*

Many hand-built monitors accumulate only incriminating signal. Every rule that
fires pushes the state up; nothing pushes it down. The only thing that lowers
the state is the decay term. Such a monitor cannot register that the agent
recovered, corrected itself, or backed out of a bad approach -- not because it
detects those events badly, but because it has no channel in which to express
them. Its sensitivity to recovery is exactly zero, and it would stay zero
against a perfectly self-correcting agent.

Worse, and this is common: a rule that *detects* self-correction ("let me
reconsider", "on second thought") and emits a positive delta on a
negative-valence dimension is sign-inverted. Noticing that the agent is fixing
itself raises the alarm.

This audit takes your extractor as a callable, runs it over real observations,
and reports the realized sign census plus which dimensions are reachable at all.

    def extract(observation, state):
        '''return [(dimension, delta), ...]'''
        ...

    print(sign_constraint_audit(extract, observations))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

ExtractorFn = Callable[[Any, Dict[str, float]], Sequence[Tuple[str, float]]]


@dataclass
class SignReport:
    verdict: str
    n_observations: int
    n_signals: int
    n_negative: int
    n_positive: int
    n_zero: int
    per_dimension: Dict[str, Dict[str, float]]
    silent_fraction: float
    monotone_dimensions: List[str] = field(default_factory=list)
    examples_negative: List[Tuple[int, str, float]] = field(default_factory=list)

    @property
    def is_sign_constrained(self) -> bool:
        return self.verdict == "SIGN_CONSTRAINED"

    def __str__(self) -> str:
        lines = [
            "sign-constraint audit",
            f"  observations     : {self.n_observations}",
            f"  signals emitted  : {self.n_signals}",
            f"  negative deltas  : {self.n_negative}",
            f"  positive deltas  : {self.n_positive}",
            f"  silent obs       : {self.silent_fraction*100:.1f}%",
            "",
            "  dimension            n     sum      min      max   monotone?",
            "  " + "-" * 60,
        ]
        for dim, d in sorted(self.per_dimension.items(),
                             key=lambda kv: -kv[1]["n"]):
            mono = "UP-ONLY" if dim in self.monotone_dimensions else "both"
            lines.append(f"  {dim:18s} {d['n']:5.0f} {d['sum']:7.2f} "
                         f"{d['min']:8.3f} {d['max']:8.3f}   {mono}")
        lines += ["", "  " + _EXPLAIN[self.verdict]]
        return "\n".join(lines)


_EXPLAIN = {
    "SIGN_CONSTRAINED": (
        "Every emitted signal is non-negative. This extractor cannot express "
        "recovery, correction, or exoneration on ANY dimension. The only thing "
        "that lowers the monitor's state is its decay term -- which means the "
        "monitor's sensitivity to de-escalation is exactly zero, structurally, "
        "and no threshold change or model upgrade affects that. If the state "
        "also decays in wall-clock, the monitor can only be drained by the "
        "agent going quiet for long enough, never by it doing the right thing."),
    "PARTIALLY_CONSTRAINED": (
        "Some dimensions are up-only. Those dimensions cannot register "
        "recovery. Check whether the dimensions your triggers actually read "
        "are among them -- a de-escalating channel that no trigger consumes is "
        "no better than none."),
    "BALANCED": (
        "The extractor emits evidence in both directions. Recovery has a "
        "channel. This does not by itself mean the monitor detects recovery "
        "well, only that it is able to."),
    "NO_SIGNALS": (
        "The extractor emitted nothing on these observations. The audit is "
        "uninformative."),
}


def sign_constraint_audit(
    extractor: ExtractorFn,
    observations: Sequence[Any],
    initial_state: Dict[str, float] | None = None,
    trigger_dimensions: Sequence[str] | None = None,
) -> SignReport:
    """Run the extractor over real observations and census the signal signs.

    `trigger_dimensions`, if given, are the dimensions your triggers actually
    read. A monitor whose only de-escalating channel is one no trigger consumes
    is still sign-constrained where it counts, and the verdict says so.
    """
    state = dict(initial_state or {})
    per_dim: Dict[str, Dict[str, float]] = {}
    neg = pos = zero = 0
    silent = 0
    examples: List[Tuple[int, str, float]] = []

    for i, obs in enumerate(observations):
        signals = list(extractor(obs, state)) or []
        if not signals:
            silent += 1
        for dim, delta in signals:
            delta = float(delta)
            d = per_dim.setdefault(dim, {"n": 0.0, "sum": 0.0,
                                         "min": float("inf"),
                                         "max": float("-inf")})
            d["n"] += 1
            d["sum"] += delta
            d["min"] = min(d["min"], delta)
            d["max"] = max(d["max"], delta)
            if delta < 0:
                neg += 1
                if len(examples) < 10:
                    examples.append((i, dim, delta))
            elif delta > 0:
                pos += 1
            else:
                zero += 1
            state[dim] = state.get(dim, 0.0) + delta

    total = neg + pos + zero
    monotone = [k for k, v in per_dim.items() if v["min"] >= 0.0]

    if total == 0:
        verdict = "NO_SIGNALS"
    elif neg == 0:
        verdict = "SIGN_CONSTRAINED"
    elif trigger_dimensions and all(
            d in monotone for d in trigger_dimensions if d in per_dim):
        verdict = "SIGN_CONSTRAINED"
    elif monotone:
        verdict = "PARTIALLY_CONSTRAINED"
    else:
        verdict = "BALANCED"

    return SignReport(
        verdict=verdict, n_observations=len(observations), n_signals=total,
        n_negative=neg, n_positive=pos, n_zero=zero, per_dimension=per_dim,
        silent_fraction=silent / len(observations) if observations else 0.0,
        monotone_dimensions=monotone, examples_negative=examples,
    )
