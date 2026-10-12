# monitor_invariance

Four checks for anyone building a runtime monitor that carries state across an
agent's episode — a strike count, a windowed event count, a decayed risk score.
Standard library only; no dependencies.

A stateful monitor makes three design choices that are rarely written down:
what its statistic forgets against (elapsed seconds or observations seen),
whether the statistic can ever fall, and how much evidence it needs before it
can alarm at all. Each choice decides whether the monitor has a usable operating
regime at the rate an agent actually acts. These checks compute the answers for
your monitor before you deploy it.

## The two quantities

No library needed:

```
N_min  = ceil(theta / a)                (CUSUM with slack k, interval h: ceil(h / (1 - k)))
r_max  = E / N_min                      E = qualifying observations per session
```

`N_min` is the evidence floor: a monitor that adds at most `a` per qualifying
observation and alarms at `theta` cannot alarm on a session that contains fewer
than `N_min` of them, whatever it spends. `r_max` is the largest 1-in-r
monitoring budget a session can afford.

They answer two different questions, and the report keeps them separate.
**Feasibility**: does a session carry at least `N_min` qualifying observations
at full budget? **Headroom**: once it does, how far can the budget be thinned?
A monitor that fails feasibility cannot be repaired by spending more; collapsing
both into one number below 1 tells a designer to buy the wrong fix.

Two caveats travel with `r_max`. It is an expectation under uniform thinning:
with bursty evidence a particular sampling offset can keep more or fewer than
`E / r` qualifying observations, so it is a design heuristic, not a guarantee.
And it is only as good as the pipeline that delivers `E`: an extractor that
matches error words in returned file contents inflates it, and a logging layer
that records only successful tool calls deflates it.

**None of this is new statistics, and we claim no priority for it.** The CUSUM
is Page (1954). That a chart cannot signal before enough evidence accumulates
is its startup behaviour, familiar enough that the fast-initial-response
scheme of Lucas and Crosier (1982) is named for giving a chart a head start
through it; the floor follows in one line from bounded increments. That
sampling rate and detection delay are one knob is the variable-sampling-interval
literature (Reynolds, Amin, Arnold and Nachlas, 1988; Reynolds, Amin and
Arnold, 1990), where average time to signal is average run length times the
sampling interval. What this package adds is the mapping to agent monitoring:
the floor tied to a monitoring budget through `r_max`, on sessions that *end* —
where a chart that would eventually signal on an unbounded stream instead never
does.

## The four checks

| question | check | a bad answer means |
|---|---|---|
| Does my statistic depend on the clock? | `cadence_invariance_test` | `CLIFF`: budget and latency changes move it between constant alarm and silence |
| Can it ever decide things improved? | `sign_constraint_audit` | `SIGN_CONSTRAINED`: it can only wait out evidence, never retire it |
| Can it fire at all on my sessions, and how much can I skip? | `design_report` | `STARVED`, or a max safe budget below 1 |
| What does a budget do to it? | `budget_invariance_test` | fire rate collapsing between r = 1 and r = 2 |

`design_report` is arithmetic. The other three run your monitor or extractor on
a sample of your own traces.

### 1. Is my monitor wall-clock or sample-time calibrated?

A monitor with internal state has to decide what that state decays against:

```
WALL-CLOCK    state *= exp(-lambda * dt_seconds)     a half-life in seconds
SAMPLE-TIME   state updated once per observation     a CUSUM, a counter
```

Wrap your monitor so it takes `(stream, dt)` and returns one truthy value per
observation, then sweep:

```python
from monitor_invariance import cadence_invariance_test

def my_monitor(stream, dt):
    return [...]                      # one decision per observation

print(cadence_invariance_test(my_monitor, my_real_traces))
```

Example output, for a wall-clock leaky integrator on 20 SWE-bench debugging
traces:

```
verdict          : CLIFF
critical interval: (15, 30] s

  dt (s)   fired    persistence %   median fires
       0    100%          100.0           18.5
       5     85%          100.0           13.5
      15     75%           90.9            6.5
      30     35%           33.3            0.0
      60      0%            0.0            0.0
```

`INVARIANT` means sample-time calibrated: cadence and budget changes cost it
sensitivity, not a regime. `CLIFF` means wall-clock calibrated: below the band
it latches on, above it it never fires, and anything that changes the spacing of
its observations moves it across that boundary without anyone editing a
threshold.

**Calibration and saturation are reported separately, on purpose.** A monitor
can be perfectly invariant and still useless because its threshold is too low.
Retuning a threshold will not remove a cliff, and recalibrating onto the
observation clock will not fix a threshold. Conflating the two is the mistake
this package exists to prevent — the first version of this code made it.

### 2. What does a monitoring budget do to it?

Monitoring every action of every agent is expensive, so deployments sample
1-in-r. For any monitor whose forgetting is indexed in seconds — an exponential
decay, or a rule of the form "at least N events in the last T seconds" — that
multiplies the elapsed time between observations by r: **observing one action
in r is exactly equivalent to the agent acting r times more slowly.**

```python
from monitor_invariance import budget_invariance_test
budget_invariance_test(my_monitor, traces, cadence=1.53)
```

On the reference instruments, over 20 SWE-bench-Verified debugging trajectories
at a measured 1.53 s cadence (sampling offset 0), the share of traces on which
the monitor fired at all:

| monitor | r=1 | r=2 | r=5 | r=10 | r=20 | r=50 |
|---|---|---|---|---|---|---|
| wall-clock accumulator | 95% | 65% | 5% | 0% | 0% | 0% |
| sample-time CUSUM | 100% | 95% | 60% | 20% | 5% | 0% |

Both degrade. Only one of them degrades by falling off a cliff. Both reach zero
eventually: a 1-in-50 budget leaves a 30-action trace with a single
observation, and no statistic saves you from that.

### 3. Can my monitor ever register that things got better?

A separate failure mode, and one no retuning fixes. Many hand-built monitors
accumulate only incriminating evidence: every rule that fires pushes the state
up, nothing pushes it down, and the only thing that lowers the state is decay
or expiry. Such a monitor cannot register that the agent recovered — not
because it detects recovery badly, but because it has no channel in which to
express it.

```python
from monitor_invariance import sign_constraint_audit
print(sign_constraint_audit(my_extractor, observations,
                            trigger_dimensions=["suspicion"]))
```

```
  dimension            n     sum      min      max   monotone?
  frustration           12    1.05    0.050    0.150   UP-ONLY
  confusion              8    1.15    0.100    0.150   UP-ONLY
  fear                   4    0.20    0.050    0.050   UP-ONLY

  verdict: SIGN_CONSTRAINED
```

`trigger_dimensions` matters: a de-escalating channel that no trigger reads is
no better than none, and the verdict accounts for that.

### 4. Can my monitor fire at all, and how much can I skip?

`design_report` computes both bounds for a leaky integrator and returns
whichever binds. **Always supply the session length**; without it the report
can compute only the clock bound, and says it is overstating.

```
N_min   = ceil(theta / a)                          evidence events needed to fire
r_max   = E / N_min                                floor bound (E = evidence per trace)
delta*  = -(H / ln 2) * ln(1 - a*p/theta)          critical interval (input rate p)
r_cliff = delta* / cadence                         clock bound
max safe 1-in-r budget = min(r_max, r_cliff)
```

```python
from monitor_invariance import design_report
print(design_report(step=0.15, rate=0.5, half_life=150, threshold=0.7,
                    cadence=1.53, trace_length=7))
```

```
  critical interval: 24.5 s
  cliff bound      : 1-in-16.0
  starvation bound : 1-in-0.7  (N_min = 5 evidence events; ~3.5 expected per trace)
  MAX SAFE BUDGET  : BELOW 1 (0.70) -- the monitor cannot reach threshold on a
                     trace this short even with nothing skipped. Binding: starvation.
  regime: STARVED -- ...
```

Same monitor, same cadence, three trace lengths:

| trace length | cliff bound | starvation bound | max safe budget |
|---|---|---|---|
| not supplied | 1-in-16.0 | — | **1-in-16.0 — overstated** |
| 32 actions | 1-in-16.0 | 1-in-3.2 | **1-in-3.2** |
| 7 actions | 1-in-16.0 | 0.7 | **below 1 — cannot fire** |

**An earlier version of this package reported only the cliff bound as "max safe
budget"** and so told its authors' own monitor it could skip 15 of every 16
actions. On a 32-action trace the real figure is about 1-in-3; on a 7-action
trace it is zero.

For a sample-time CUSUM the floor is `cusum_evidence_floor(h, k)`; for any
accumulator, `evidence_floor(threshold, step)`. If `a*p >= theta` the cliff
formula returns `None`: the input outruns the decay at every interval, and the
monitor is a latch rather than a detector.

**The floor needs no empirical validation — it is arithmetic.** An accumulator
that needs 5 evidence events cannot fire on 4. The empirical content is how many
real sessions fall below the floor, and that depends on what counts as evidence.
On 16 sessions from a production agent harness, counting only shell commands
that returned error text, 15 of 16 contain fewer than 5, so a 5-event
accumulator cannot fire on them even at full budget; across five evidence
definitions the figure is 14–15 of 16, except under one inflated by
source-file contents, where it is 6 of 16 (Paper 3, Appendix A).

**A lower floor helps only if evidence arrives close enough together.** A CUSUM
with `k = 0.1, h = 1.0` needs 2 events, but only if at most 8 clean observations
separate them. On the same 16 sessions it fired on the 8 with two or more
execution failures — in each of them at least one consecutive pair fell within
8 — and held 50% → 12% → 5% at full, half and quarter budget, against
6% → 0% → 0% for the best-case accumulator. The design rule is two-parameter:
minimise `N_min`, then check the statistic's evidence window against the spacing
of qualifying observations in your own sessions.

**Validation of the cliff formula.** Against 20 real agent traces, the closed
form for `delta*` lands inside the measured critical interval on 13 of 20
traces, and every miss is exactly one grid bin. It has no fitted parameters, so
that absolute agreement is a test of the rule rather than a fit to it. Do not
quote a rank correlation for it: `delta*` is strictly monotone in the error rate
once `a`, `H` and `theta` are fixed, so any rank correlation with the measured
interval equals the error rate's own and tests nothing about the formula. What
the formula buys over the error rate is absolute calibration — in the example
above, a cliff at 24.5 s for an input rate of 0.5.

## Install and test

```
pip install .
python -m pytest
```

The suite (129 tests) checks the budget–cadence identity against an explicit
simulation driven by real timestamps, for both clock-indexed forms (exponential
decay and the "N events in T seconds" window), rather than through the code
path that assumes it; that `design_report` says `STARVED` exactly when the
expected evidence is below the floor; and the sign audit's verdicts.

## Citing

See `CITATION.cff`. The checks are described in:

Modgil, M. (2026). *Clocks, Counts, and Evidence Floors in Stateful Runtime
Monitoring of AI Agents.* Forthcoming.

Modgil, M. (2026). *Bistable by Construction: Wall-Clock-Calibrated State
Monitors Have No Moment-Detection Regime at Agent Cadence.* arXiv:2606.19386.

## References

Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41(1–2),
100–115. doi:10.1093/biomet/41.1-2.100

Lucas, J. M. and Crosier, R. B. (1982). Fast initial response for CUSUM
quality-control schemes: give your CUSUM a head start. *Technometrics* 24(3),
199–205. doi:10.1080/00401706.1982.10487759

Reynolds, M. R., Amin, R. W., Arnold, J. C. and Nachlas, J. A. (1988). X̄ charts
with variable sampling intervals. *Technometrics* 30(2), 181–192.
doi:10.1080/00401706.1988.10488366

Reynolds, M. R., Amin, R. W. and Arnold, J. C. (1990). CUSUM charts with
variable sampling intervals. *Technometrics* 32(4), 371–384.
doi:10.1080/00401706.1990.10484721

## License

Apache-2.0. See `LICENSE`.
