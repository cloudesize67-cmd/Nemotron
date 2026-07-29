"""Seed candidate for the evolutionary search.

This is a deliberately naive first guess at the BMV relative-phase formula:
it gets the linear scaling in G/m1/m2/T/hbar right but uses the wrong power
of delta_x and d (a plausible "dimensional analysis by eyeballing" mistake).
The evolutionary loop's job is to mutate this into the correct quadratic
delta_x / inverse-cube d scaling law derived in ``physics_reference``.

# EVOLVE-BLOCK-START
"""

import math


def solve(params: dict) -> dict:
    G = params["G"]
    m1 = params["m1"]
    m2 = params["m2"]
    delta_x = params["delta_x"]
    d = params["d"]
    T = params["T"]
    hbar = params["hbar"]

    phase = G * m1 * m2 * delta_x * T / (hbar * d**2)
    concurrence = abs(math.sin(phase))
    return {"phase": phase, "concurrence": concurrence}


# EVOLVE-BLOCK-END
