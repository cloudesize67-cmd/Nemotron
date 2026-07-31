import sympy as sp
from bmv_evolve import physics_reference as pr


def test_relative_phase_matches_known_closed_form():
    phi = pr.relative_phase_symbolic()
    expected = 2 * pr.G * pr.m1 * pr.m2 * pr.delta_x**2 * pr.T / (pr.hbar * pr.d**3)
    assert sp.simplify(phi - expected) == 0


def test_scaling_exponents_match_expected():
    phi = pr.relative_phase_symbolic()
    exponents = pr.scaling_exponents(phi)
    for name, expected in pr.EXPECTED_SCALING_EXPONENTS.items():
        assert exponents[name] == expected, f"{name}: got {exponents[name]}, expected {expected}"


def test_concurrence_bounds():
    assert pr.concurrence_symbolic(sp.Integer(0)) == 0
    assert sp.simplify(pr.concurrence_symbolic(sp.pi / 2) - 1) == 0


def test_numeric_result_positive_and_finite():
    params = {
        "G": pr.G_SI,
        "m1": 1e-14,
        "m2": 1e-14,
        "delta_x": 1e-6,
        "d": 1e-4,
        "T": 1.0,
        "hbar": pr.HBAR_SI,
    }
    result = pr.numeric_result(params)
    assert result.phase > 0
    assert 0.0 <= result.concurrence <= 1.0
