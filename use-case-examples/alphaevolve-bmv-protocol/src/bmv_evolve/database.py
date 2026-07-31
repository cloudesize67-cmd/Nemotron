"""A minimal program database for the evolutionary loop.

Real FunSearch/AlphaEvolve maintain island populations with diversity
pressure; this example keeps just enough to be a legitimate (if small)
evolutionary search: a bounded population sorted by score, with
score-weighted parent sampling so the search still explores rather than
always hill-climbing from the single best program.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .evaluator import EvalResult


@dataclass
class Program:
    id: int
    code: str
    result: EvalResult
    generation: int
    parent_id: int | None = None


@dataclass
class ProgramDatabase:
    capacity: int = 16
    _programs: list[Program] = field(default_factory=list)
    _next_id: int = 0

    def add(self, code: str, result: EvalResult, generation: int, parent_id: int | None) -> Program:
        program = Program(id=self._next_id, code=code, result=result, generation=generation, parent_id=parent_id)
        self._next_id += 1
        self._programs.append(program)
        self._programs.sort(key=lambda p: p.result.score, reverse=True)
        del self._programs[self.capacity :]
        return program

    def sample_parent(self, rng: random.Random) -> Program:
        if not self._programs:
            raise ValueError("cannot sample from an empty database")
        weights = [max(p.result.score, 1e-3) for p in self._programs]
        return rng.choices(self._programs, weights=weights, k=1)[0]

    def best(self) -> Program:
        if not self._programs:
            raise ValueError("database is empty")
        return self._programs[0]

    def __len__(self) -> int:
        return len(self._programs)

    def history(self) -> list[Program]:
        return list(self._programs)
