"""The LLM "actor" that proposes mutated candidate programs.

AlphaEvolve/FunSearch drive the search with an LLM that reads the current
best program(s) plus evaluator feedback and proposes an improved rewrite.
This module defines that interface (``Actor``) and two implementations:

- ``NemotronActor``: calls a Nemotron model through an OpenAI-compatible
  endpoint (NVIDIA's ``build.nvidia.com`` / NIM API by default). This is
  the "real" actor and is what you want for actually evolving the BMV
  solver.
- ``MockActor``: a small deterministic local mutator with no network
  dependency, used as a fallback when no API key is configured and by the
  test suite, so the evolutionary loop's *mechanics* are exercisable and
  testable offline. It is not meant to discover new physics -- only to
  exercise the database/loop plumbing.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

MUTATION_PROMPT_TEMPLATE = """\
You are improving a Python function that computes the leading-order relative
phase and entanglement (concurrence) for the Bose-Marletto-Vedral (BMV)
quantum-gravity witness protocol.

The function receives a `params` dict with keys G, m1, m2, delta_x, d, T,
hbar (SI units) and must return {{"phase": float, "concurrence": float}}.

The true physics: two test masses, each split into a spatial superposition
of two paths offset by `delta_x`, held at mean separation `d` for a hold
time `T`. The relative Newtonian phase between the correlated and
anti-correlated interferometer branches follows a power law in each
parameter. Your goal is to make `solve()` match that power law exactly.

Current program:
```python
{code}
```

Evaluator feedback on the current program:
{feedback}

Rewrite the ENTIRE program (imports included) with an improved `solve()`
that better matches the expected scaling exponents and numeric values.
Keep the function signature `def solve(params: dict) -> dict:` unchanged.
Respond with ONLY a single python code block, no prose.
"""


class Actor(ABC):
    """Proposes a mutated program given a parent program and its evaluation feedback."""

    @abstractmethod
    def propose(self, parent_code: str, feedback: str) -> str:
        raise NotImplementedError


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


class NemotronActor(Actor):
    """Calls a Nemotron model through an OpenAI-compatible chat completions endpoint.

    Configure via environment variables:
      - ``NVIDIA_API_KEY``: API key for https://integrate.api.nvidia.com/v1
        (get one at https://build.nvidia.com).
      - ``BMV_LLM_BASE_URL`` (optional): override the API base URL, e.g. to
        point at OpenRouter or a self-hosted NIM instead.
      - ``BMV_LLM_MODEL`` (optional): override the model name.
    """

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"

    def __init__(self, model: str | None = None, base_url: str | None = None, temperature: float = 0.7):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise RuntimeError(
                "NemotronActor requires the 'openai' package. Install the example's "
                "extras with `uv sync --extra llm` or `pip install openai`."
            ) from exc

        api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set NVIDIA_API_KEY (or OPENAI_API_KEY) to use NemotronActor.")

        self.model = model or os.environ.get("BMV_LLM_MODEL", self.DEFAULT_MODEL)
        base_url = base_url or os.environ.get("BMV_LLM_BASE_URL", self.DEFAULT_BASE_URL)
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def propose(self, parent_code: str, feedback: str) -> str:
        prompt = MUTATION_PROMPT_TEMPLATE.format(code=parent_code, feedback=feedback)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return _extract_code_block(response.choices[0].message.content or "")


class MockActor(Actor):
    """Deterministic, network-free mutator used offline and in tests.

    Cycles through a small library of hand-written patches that nudge the
    seed program's power-law exponents toward the correct values, so a run
    of the evolutionary loop with this actor is fully reproducible.
    """

    _PATCHES = [
        # Full-line (find, replace) pairs applied in sequence across generations,
        # each keyed on the exact line left behind by the previous patch.
        (
            "phase = G * m1 * m2 * delta_x * T / (hbar * d**2)",
            "phase = G * m1 * m2 * delta_x**2 * T / (hbar * d**2)",
        ),
        (
            "phase = G * m1 * m2 * delta_x**2 * T / (hbar * d**2)",
            "phase = G * m1 * m2 * delta_x**2 * T / (hbar * d**3)",
        ),
        (
            "phase = G * m1 * m2 * delta_x**2 * T / (hbar * d**3)",
            "phase = 2 * G * m1 * m2 * delta_x**2 * T / (hbar * d**3)",
        ),
    ]

    def propose(self, parent_code: str, feedback: str) -> str:
        for find, replace in self._PATCHES:
            if find in parent_code:
                return parent_code.replace(find, replace)
        return parent_code


def get_actor(model: str | None = None, base_url: str | None = None) -> Actor:
    """Return a NemotronActor if credentials are configured, else fall back to MockActor."""
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return NemotronActor(model=model, base_url=base_url)
    return MockActor()
