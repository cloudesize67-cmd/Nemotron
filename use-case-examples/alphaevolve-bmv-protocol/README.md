# AlphaEvolve-Style Evolutionary Pipeline for the BMV Protocol

An end-to-end example of an **AlphaEvolve/FunSearch-style evolutionary
coding pipeline** — an LLM "actor" that proposes candidate programs, and an
automated evaluator that scores them — applied to the
**Bose-Marletto-Vedral (BMV) protocol**, a proposed table-top experiment for
testing whether gravity must be quantized. It uses a **Nemotron model as the
LLM actor**, demonstrating how Nemotron slots into this class of
"evolve-a-scientific-solver" agentic workflow.

This is a worked demo of the pipeline mechanics, not a research claim: the
evaluator checks a well-established leading-order result from the BMV
literature, and the default offline mode uses a scripted mock actor rather
than a real model call.

## Background

### The physics: BMV protocol

The BMV protocol (Bose et al. 2017, *"Spin Entanglement Witness for Quantum
Gravity"*; Marletto & Vedral 2017, *"Gravitationally Induced Entanglement
between Two Massive Particles is Sufficient Evidence of Quantum Effects in
Gravity"*) proposes splitting two test masses `m1` and `m2` — mean
separation `d` — each into a spatial superposition of two paths offset by
`delta_x`, and holding them for a time `T`. The four path combinations pick
up different Newtonian gravitational phases; the branch that keeps the
masses at their original separation interferes differently from the branch
that briefly moves them closer or farther apart. The resulting relative
phase seeds entanglement between the two mass "qubits."

The argument driving the physics debate: if gravity were a strictly
classical, LOCC-style mediator, it could **not** entangle two initially
independent quantum systems. Observing a nonzero entanglement witness would
therefore be evidence that the gravitational field itself carries quantum
degrees of freedom — the "classical vs. quantum gravity" question this
example's evaluator is built around.

This example's `physics_reference.py` derives the leading-order relative
phase symbolically with SymPy (a Taylor expansion of the Newtonian
potential around `delta_x = 0` — the "cross-propagator" term coupling the
two mass superpositions), recovering the standard scaling law:

```
phi ~ G * m1 * m2 * delta_x**2 * T / (hbar * d**3)
```

and the associated entanglement witness, concurrence `C = |sin(phi)|`.

### The architecture: AlphaEvolve / FunSearch

[AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
and its predecessor [FunSearch](https://github.com/google-deepmind/funsearch)
pair an LLM with an automated evaluator in an evolutionary loop:

1. Maintain a database of candidate programs, scored by the evaluator.
2. Sample a parent program (score-weighted).
3. Ask the LLM to rewrite/improve it, given the parent code and the
   evaluator's feedback.
4. Score the child program and insert it back into the database.
5. Repeat, tracking the best program found so far.

[OpenEvolve](https://github.com/codelion/openevolve) is a popular
independent open-source reimplementation of the AlphaEvolve architecture.
This example implements the same loop directly (see `evolve.py`) so it has
no external framework dependency beyond an LLM client.

## What's in this example

| File | Role |
|---|---|
| `src/bmv_evolve/physics_reference.py` | Symbolic ground truth (SymPy): derives the relative phase, scaling exponents, and concurrence formula. |
| `src/bmv_evolve/evaluator.py` | The evaluator: runs a candidate's `solve()` against random parameter sets and a scaling-law probe, scores numeric agreement and exponent recovery. |
| `src/bmv_evolve/_runner.py` | Subprocess entry point that executes candidate code in isolation, with a timeout. |
| `src/bmv_evolve/seed_program.py` | The initial (deliberately wrong-exponent) candidate the search starts from. |
| `src/bmv_evolve/llm_actor.py` | The LLM actor interface: `NemotronActor` (real model calls) and `MockActor` (offline, deterministic, no network). |
| `src/bmv_evolve/database.py` | A small score-weighted program population. |
| `src/bmv_evolve/evolve.py` | The evolutionary loop and CLI entry point. |
| `tests/` | Unit tests for the physics reference, evaluator, and the mock evolutionary run. |

## Running it

```bash
cd use-case-examples/alphaevolve-bmv-protocol
uv sync --extra llm --extra dev   # or: pip install -e ".[llm,dev]"
```

### Offline (no API key, deterministic)

```bash
uv run python -m bmv_evolve.evolve --iterations 10
```

With no `NVIDIA_API_KEY` set, `get_actor()` falls back to `MockActor`, a
small scripted patch library that fixes the seed program's exponents step
by step. This exercises the full database/evaluation loop without any
network dependency — useful for CI and for understanding the mechanics
before spending API calls.

### With a Nemotron model as the actor

```bash
export NVIDIA_API_KEY=nvapi-...          # from https://build.nvidia.com
uv run python -m bmv_evolve.evolve \
    --iterations 30 \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --out-dir bmv_evolve_out
```

Each generation, the Nemotron model receives the current parent program plus
the evaluator's feedback (numeric match score, scaling-law match score, and
which parameter exponents were measured vs. expected) and proposes a full
rewrite of `solve()`. `--base-url` lets you point at a different
OpenAI-compatible endpoint (e.g. OpenRouter, or a self-hosted NIM).

Outputs land in `--out-dir` (default `bmv_evolve_out/`):
`best_program.py` (the top-scoring candidate found) and `history.json`
(per-generation scores).

## The evaluator, in detail

A candidate program must implement:

```python
def solve(params: dict) -> dict:
    # params: G, m1, m2, delta_x, d, T, hbar (SI units)
    return {"phase": ..., "concurrence": ...}
```

`evaluate()` in `evaluator.py` scores it on two independent axes, averaged
50/50:

- **Numeric match**: relative error on `phase` and absolute error on
  `concurrence` against `physics_reference`, averaged over randomly sampled
  physically-plausible parameter sets.
- **Scaling-law match**: perturbs each of the seven parameters individually
  about a fixed baseline and measures the empirical power-law exponent from
  the candidate's own output, compared against the expected exponents
  (`G^1 m1^1 m2^1 delta_x^2 T^1 d^-3 hbar^-1`). This is what actually tests
  whether the search recovered the right *physics*, not just a numeric fit
  at a handful of points.

A candidate that fails to import, raises, or times out (8s default, run in
an isolated subprocess via `_runner.py`) scores `0.0`.

## Extending this to a harder problem

The pieces here generalize beyond the leading-order estimate:

- Add relativistic retardation corrections to `physics_reference.py`
  (the interaction Hamiltonian used here assumes an instantaneous, static
  Newtonian potential during the hold time).
- Extend the evaluator to also grade a decoherence budget (how large an
  environmental decoherence rate the setup can tolerate while still keeping
  `phi = O(1)`).
- Swap in a full quantized-field treatment of the mass-mass coupling as a
  second reference model, and have the evaluator score candidates on how
  well they distinguish the classical-mediator prediction (no entanglement
  possible) from the quantized-field prediction — the actual "classical vs.
  quantum gravity" discriminator the BMV literature is aiming at.

## References

- S. Bose et al., *"Spin Entanglement Witness for Quantum Gravity,"*
  Phys. Rev. Lett. 119, 240401 (2017).
- C. Marletto & V. Vedral, *"Gravitationally Induced Entanglement between
  Two Massive Particles is Sufficient Evidence of Quantum Effects in
  Gravity,"* Phys. Rev. Lett. 119, 240402 (2017).
- B. Romera-Paredes et al., *"Mathematical discoveries from program search
  with large language models" (FunSearch)*, Nature 625, 468-475 (2024).
  [github.com/google-deepmind/funsearch](https://github.com/google-deepmind/funsearch)
- Google DeepMind, *"AlphaEvolve: A coding agent for scientific and
  algorithmic discovery,"* 2025.
- [OpenEvolve](https://github.com/codelion/openevolve) — independent
  open-source reimplementation of the AlphaEvolve architecture.
