"""The physics evaluator: the strict judge of a candidate program's mathematical accuracy.

Each candidate program must define::

    def solve(params: dict) -> dict:
        # params has keys: G, m1, m2, delta_x, d, T, hbar (all SI units)
        return {"phase": <float>, "concurrence": <float>}

The evaluator scores a candidate on two independent axes:

1. **Numeric agreement** with the symbolic reference (``physics_reference``)
   across randomly sampled, physically plausible parameter sets.
2. **Scaling-law recovery**: whether perturbing each parameter individually
   about a fixed baseline reproduces the correct power-law exponent
   (``G^1 m1^1 m2^1 delta_x^2 T^1 d^-3 hbar^-1``), which is the actual
   physics content an evolutionary search over this problem is meant to
   discover rather than curve-fit.

A candidate that fails to import, times out, or raises on every trial
scores 0.0.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from . import physics_reference

_RUNNER_PATH = Path(__file__).parent / "_runner.py"

DEFAULT_TIMEOUT_S = 8.0
PERTURB_FACTOR = 3.0
PHASE_TOLERANCE = 0.05
CONCURRENCE_TOLERANCE = 0.05
EXPONENT_TOLERANCE = 0.25

BASELINE_PARAMS = {
    "G": physics_reference.G_SI,
    "m1": 1e-14,
    "m2": 1e-14,
    "delta_x": 1e-6,
    "d": 1e-4,
    "T": 1.0,
    "hbar": physics_reference.HBAR_SI,
}


@dataclass
class EvalResult:
    score: float
    numeric_score: float
    scaling_score: float
    n_trials: int
    n_errors: int
    errors: list[str] = field(default_factory=list)
    empirical_exponents: dict[str, float] = field(default_factory=dict)

    def feedback(self) -> str:
        """Human/LLM-readable summary used to steer the next mutation."""
        lines = [
            f"score={self.score:.3f} (numeric_match={self.numeric_score:.3f}, "
            f"scaling_law_match={self.scaling_score:.3f}), "
            f"{self.n_errors}/{self.n_trials} trials errored."
        ]
        if self.empirical_exponents:
            exp_lines = []
            for name, expected in physics_reference.EXPECTED_SCALING_EXPONENTS.items():
                got = self.empirical_exponents.get(name)
                got_str = f"{got:.2f}" if got is not None else "N/A"
                exp_lines.append(f"  {name}: expected exponent {expected}, empirically measured {got_str}")
            lines.append("Scaling exponents (phase should scale as param**exponent):")
            lines.extend(exp_lines)
        if self.errors:
            lines.append("Sample errors:")
            lines.extend(f"  - {e}" for e in self.errors[:5])
        return "\n".join(lines)


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-300)


def _sample_random_params(rng: random.Random, n: int) -> list[dict[str, float]]:
    trials = []
    for _ in range(n):
        delta_x = 10 ** rng.uniform(-7, -4)
        d = delta_x * 10 ** rng.uniform(0.5, 2.5)
        trials.append(
            {
                "G": physics_reference.G_SI,
                "m1": 10 ** rng.uniform(-17, -13),
                "m2": 10 ** rng.uniform(-17, -13),
                "delta_x": delta_x,
                "d": d,
                "T": 10 ** rng.uniform(-1, 1),
                "hbar": physics_reference.HBAR_SI,
            }
        )
    return trials


def _run_candidate(program_path: str, trials: list[dict], timeout_s: float) -> list[dict] | None:
    try:
        completed = subprocess.run(
            [sys.executable, str(_RUNNER_PATH), str(program_path), json.dumps(trials)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None

    if completed.returncode != 0 or not completed.stdout.strip():
        return None

    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return None


def evaluate(
    program_path: str,
    num_random_trials: int = 8,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    seed: int = 0,
) -> EvalResult:
    rng = random.Random(seed)
    random_trials = _sample_random_params(rng, num_random_trials)

    param_names = list(physics_reference.EXPECTED_SCALING_EXPONENTS)
    scaling_trials = [dict(BASELINE_PARAMS)] + [
        {**BASELINE_PARAMS, name: BASELINE_PARAMS[name] * PERTURB_FACTOR} for name in param_names
    ]
    all_trials = random_trials + scaling_trials

    results = _run_candidate(program_path, all_trials, timeout_s)
    if results is None or len(results) != len(all_trials):
        return EvalResult(
            score=0.0,
            numeric_score=0.0,
            scaling_score=0.0,
            n_trials=len(all_trials),
            n_errors=len(all_trials),
            errors=["candidate program failed to run (timeout, crash, or malformed output)"],
        )

    random_results = results[:num_random_trials]
    scaling_results = results[num_random_trials:]
    errors: list[str] = []

    numeric_scores = []
    for params, res in zip(random_trials, random_results):
        if not res.get("ok"):
            numeric_scores.append(0.0)
            errors.append(res.get("error", "unknown error"))
            continue
        ref = physics_reference.numeric_result(params)
        phase_err = _relative_error(res["phase"], ref.phase)
        conc_err = abs(res["concurrence"] - ref.concurrence)
        numeric_scores.append(
            0.5 * math.exp(-phase_err / PHASE_TOLERANCE) + 0.5 * math.exp(-conc_err / CONCURRENCE_TOLERANCE)
        )
    numeric_score = mean(numeric_scores) if numeric_scores else 0.0

    baseline_res = scaling_results[0] if scaling_results else None
    scaling_scores = []
    empirical_exponents: dict[str, float] = {}
    if baseline_res is not None and baseline_res.get("ok") and baseline_res["phase"] > 0:
        base_phase = baseline_res["phase"]
        for name, res in zip(param_names, scaling_results[1:]):
            if not res.get("ok") or res["phase"] <= 0:
                scaling_scores.append(0.0)
                errors.append(res.get("error", f"non-positive phase while perturbing {name}"))
                continue
            empirical_exp = math.log(res["phase"] / base_phase) / math.log(PERTURB_FACTOR)
            empirical_exponents[name] = empirical_exp
            expected_exp = float(physics_reference.EXPECTED_SCALING_EXPONENTS[name])
            scaling_scores.append(math.exp(-abs(empirical_exp - expected_exp) / EXPONENT_TOLERANCE))
    else:
        errors.append(baseline_res.get("error", "baseline trial failed") if baseline_res else "no baseline result")
    scaling_score = mean(scaling_scores) if scaling_scores else 0.0

    n_errors = sum(1 for r in results if not r.get("ok"))
    score = 0.5 * numeric_score + 0.5 * scaling_score

    return EvalResult(
        score=score,
        numeric_score=numeric_score,
        scaling_score=scaling_score,
        n_trials=len(all_trials),
        n_errors=n_errors,
        errors=errors[:5],
        empirical_exponents=empirical_exponents,
    )
