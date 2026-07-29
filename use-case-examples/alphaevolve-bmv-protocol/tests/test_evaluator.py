import textwrap

import pytest
from bmv_evolve.evaluator import evaluate

CORRECT_PROGRAM = textwrap.dedent(
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

        phase = 2 * G * m1 * m2 * delta_x**2 * T / (hbar * d**3)
        concurrence = abs(math.sin(phase))
        return {"phase": phase, "concurrence": concurrence}
    """
)

WRONG_EXPONENT_PROGRAM = textwrap.dedent(
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

        # Wrong powers of delta_x and d.
        phase = G * m1 * m2 * delta_x * T / (hbar * d**2)
        concurrence = abs(math.sin(phase))
        return {"phase": phase, "concurrence": concurrence}
    """
)

ALWAYS_ZERO_PROGRAM = textwrap.dedent(
    """
    def solve(params: dict) -> dict:
        return {"phase": 0.0, "concurrence": 0.0}
    """
)

BROKEN_PROGRAM = textwrap.dedent(
    """
    def solve(params: dict) -> dict:
        raise RuntimeError("intentionally broken")
    """
)

INFINITE_LOOP_PROGRAM = textwrap.dedent(
    """
    def solve(params: dict) -> dict:
        while True:
            pass
    """
)


def _write(tmp_path, name, code):
    path = tmp_path / name
    path.write_text(code)
    return path


def test_correct_program_scores_high(tmp_path):
    path = _write(tmp_path, "correct.py", CORRECT_PROGRAM)
    result = evaluate(str(path), num_random_trials=6, seed=1)
    assert result.score > 0.9, result.feedback()
    assert result.numeric_score > 0.9
    assert result.scaling_score > 0.9
    assert result.n_errors == 0


def test_wrong_exponent_program_scores_lower_than_correct(tmp_path):
    correct_result = evaluate(str(_write(tmp_path, "correct.py", CORRECT_PROGRAM)), num_random_trials=6, seed=1)
    wrong_result = evaluate(str(_write(tmp_path, "wrong.py", WRONG_EXPONENT_PROGRAM)), num_random_trials=6, seed=1)

    assert wrong_result.score < correct_result.score
    assert wrong_result.numeric_score < correct_result.numeric_score
    # The exponents on delta_x and d are specifically wrong in this program.
    assert wrong_result.empirical_exponents["delta_x"] != pytest.approx(2, abs=0.1)
    assert wrong_result.empirical_exponents["d"] != pytest.approx(-3, abs=0.1)


def test_always_zero_program_scores_low(tmp_path):
    path = _write(tmp_path, "zero.py", ALWAYS_ZERO_PROGRAM)
    result = evaluate(str(path), num_random_trials=6, seed=1)
    assert result.score < 0.3, result.feedback()
    assert result.scaling_score == 0.0


def test_broken_program_scores_zero(tmp_path):
    path = _write(tmp_path, "broken.py", BROKEN_PROGRAM)
    result = evaluate(str(path), num_random_trials=4, seed=1)
    assert result.score == 0.0
    assert result.n_errors == result.n_trials
    assert result.errors


def test_missing_solve_function_scores_zero(tmp_path):
    path = _write(tmp_path, "nosolve.py", "x = 1\n")
    result = evaluate(str(path), num_random_trials=4, seed=1)
    assert result.score == 0.0


def test_infinite_loop_program_times_out_and_scores_zero(tmp_path):
    path = _write(tmp_path, "loop.py", INFINITE_LOOP_PROGRAM)
    result = evaluate(str(path), num_random_trials=2, timeout_s=1.5, seed=1)
    assert result.score == 0.0
