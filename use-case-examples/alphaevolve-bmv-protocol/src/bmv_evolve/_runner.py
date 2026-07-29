"""Subprocess entry point that executes a candidate program's solve() in isolation.

Invoked as ``python _runner.py <candidate_path> <trials_json>``. Kept as a
standalone script (no imports from the rest of this package) so it can be
launched as a fresh, short-lived subprocess -- this bounds the blast radius
of LLM-generated candidate code (infinite loops, crashes, stray prints)
away from the evolutionary loop's own process.
"""

from __future__ import annotations

import importlib.util
import json
import sys


def main() -> None:
    candidate_path, trials_json = sys.argv[1], sys.argv[2]
    trials = json.loads(trials_json)

    spec = importlib.util.spec_from_file_location("bmv_candidate", candidate_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    results = []
    for params in trials:
        try:
            out = module.solve(dict(params))
            results.append(
                {
                    "ok": True,
                    "phase": float(out["phase"]),
                    "concurrence": float(out["concurrence"]),
                }
            )
        except Exception as exc:  # noqa: BLE001 - candidate code is arbitrary and untrusted
            results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    print(json.dumps(results))


if __name__ == "__main__":
    main()
