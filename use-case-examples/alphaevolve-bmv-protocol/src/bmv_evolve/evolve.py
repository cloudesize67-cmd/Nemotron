"""The evolutionary loop: AlphaEvolve/FunSearch mechanics wired to the BMV evaluator.

Each generation:
  1. Sample a parent program from the database (score-weighted).
  2. Ask the LLM actor to propose an improved rewrite, given the parent's
     code and the evaluator's feedback on it.
  3. Score the child with the physics evaluator.
  4. Insert the child into the database, which keeps the top-``population_size``
     programs by score.

Run it directly:

    uv run python -m bmv_evolve.evolve --iterations 30

By default this uses ``MockActor`` (no network calls). Set ``NVIDIA_API_KEY``
to switch to a real Nemotron model as the actor.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import tempfile
from pathlib import Path

from .database import ProgramDatabase
from .evaluator import evaluate
from .llm_actor import Actor, get_actor

DEFAULT_SEED_PROGRAM = Path(__file__).parent / "seed_program.py"


def run_evolution(
    seed_path: Path,
    actor: Actor,
    iterations: int = 20,
    population_size: int = 12,
    num_random_trials: int = 8,
    rng_seed: int = 0,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> ProgramDatabase:
    rng = random.Random(rng_seed)
    db = ProgramDatabase(capacity=population_size)

    with contextlib.ExitStack() as stack:
        if out_dir is not None:
            candidates_dir = Path(out_dir) / "candidates"
            candidates_dir.mkdir(parents=True, exist_ok=True)
        else:
            candidates_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="bmv_evolve_")))

        seed_code = Path(seed_path).read_text()
        seed_file = candidates_dir / "candidate_0.py"
        seed_file.write_text(seed_code)
        seed_result = evaluate(str(seed_file), num_random_trials=num_random_trials, seed=rng_seed)
        db.add(seed_code, seed_result, generation=0, parent_id=None)
        if verbose:
            print(f"[gen 0] seed score={seed_result.score:.3f}")

        for gen in range(1, iterations + 1):
            parent = db.sample_parent(rng)
            child_code = actor.propose(parent.code, parent.result.feedback())

            child_file = candidates_dir / f"candidate_{gen}.py"
            child_file.write_text(child_code)
            child_result = evaluate(str(child_file), num_random_trials=num_random_trials, seed=rng_seed + gen)

            program = db.add(child_code, child_result, generation=gen, parent_id=parent.id)
            if verbose:
                print(
                    f"[gen {gen}] parent={parent.id} -> id={program.id} "
                    f"score={child_result.score:.3f} best={db.best().result.score:.3f}"
                )

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "best_program.py").write_text(db.best().code)
            history = [
                {
                    "id": p.id,
                    "generation": p.generation,
                    "parent_id": p.parent_id,
                    "score": p.result.score,
                    "numeric_score": p.result.numeric_score,
                    "scaling_score": p.result.scaling_score,
                }
                for p in sorted(db.history(), key=lambda p: p.id)
            ]
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    return db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PROGRAM)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--num-random-trials", type=int, default=8)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--model", type=str, default=None, help="Nemotron model name (NemotronActor only)")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--out-dir", type=Path, default=Path("bmv_evolve_out"))
    args = parser.parse_args()

    actor = get_actor(model=args.model, base_url=args.base_url)
    print(f"Using actor: {type(actor).__name__}")

    db = run_evolution(
        seed_path=args.seed_path,
        actor=actor,
        iterations=args.iterations,
        population_size=args.population_size,
        num_random_trials=args.num_random_trials,
        rng_seed=args.rng_seed,
        out_dir=args.out_dir,
    )

    best = db.best()
    print(f"\nBest program (id={best.id}, generation={best.generation}, score={best.result.score:.3f}):")
    print(best.code)
    print(best.result.feedback())


if __name__ == "__main__":
    main()
