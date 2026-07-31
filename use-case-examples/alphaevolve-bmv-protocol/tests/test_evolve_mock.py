from bmv_evolve.evolve import DEFAULT_SEED_PROGRAM, run_evolution
from bmv_evolve.llm_actor import MockActor


def test_mock_evolution_improves_on_seed(tmp_path):
    actor = MockActor()
    db = run_evolution(
        seed_path=DEFAULT_SEED_PROGRAM,
        actor=actor,
        iterations=5,
        population_size=8,
        num_random_trials=4,
        rng_seed=42,
        out_dir=tmp_path,
        verbose=False,
    )

    seed_program = [p for p in db.history() if p.generation == 0][0]
    best = db.best()

    assert len(db) >= 2
    assert best.result.score >= seed_program.result.score
    # The mock actor's patch library fixes the exponents, so evolution
    # should reach a near-perfect score within 5 generations.
    assert best.result.score > 0.9, best.result.feedback()

    assert (tmp_path / "best_program.py").exists()
    assert (tmp_path / "history.json").exists()
