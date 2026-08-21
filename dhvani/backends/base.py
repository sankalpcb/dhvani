"""Backend protocol plus the record/replay wrapper.

Two rules, both fail-closed:
  1. Replay never falls back to live on a cache miss.
  2. Live calls check the spend ledger before spending anything, and record
     the spend BEFORE the paid call executes (not after) — see RULING below.

RULING (overrides an earlier draft of this module): the live path records
spend before invoking the inner backend, not after. If record_spend() ran
after inner.transcribe(), a crash between the API call and the record_spend
call would leave money spent but unrecorded, so total_spend() would
under-count and the USD 20 ceiling could be breached on restart. Recording
first is pessimistic accounting — a failed call over-counts, which fails
safe. Preserving the ceiling matters more than perfectly accurate cost
attribution.
"""

import json
import os
from typing import Literal, Protocol, runtime_checkable

Mode = Literal["record", "replay", "live"]


class FixtureMissing(RuntimeError):
    """Replay mode was asked for a segment with no recorded fixture."""


@runtime_checkable
class Backend(Protocol):
    name: str

    def cost_per_call(self, segment) -> float: ...

    def transcribe(self, segment) -> dict: ...


class Recorded:
    """Wraps a Backend with record/replay and budget enforcement."""

    def __init__(self, inner: Backend, mode: Mode, fixture_dir: str, store=None):
        if mode not in ("record", "replay", "live"):
            raise ValueError(f"unknown mode: {mode}")
        self.inner = inner
        self.mode = mode
        self.fixture_dir = fixture_dir
        self.store = store
        self.name = inner.name

    def _path(self, segment) -> str:
        return os.path.join(self.fixture_dir, self.inner.name, f"{segment.segment_id}.json")

    def cost_per_call(self, segment) -> float:
        if self.mode == "replay":
            return 0.0
        return self.inner.cost_per_call(segment)

    def transcribe(self, segment) -> dict:
        if self.mode == "replay":
            path = self._path(segment)
            if not os.path.exists(path):
                raise FixtureMissing(
                    f"no fixture for {segment.segment_id} at {path}. "
                    f"Re-run in record mode; replay never falls back to live."
                )
            with open(path) as fh:
                return json.load(fh)

        cost = self.inner.cost_per_call(segment)
        if self.store is not None:
            self.store.check_budget(cost)
            # Record spend BEFORE the paid call, not after. A crash inside
            # inner.transcribe() must still leave the spend recorded, so
            # total_spend() can never under-count and the ceiling can never
            # be breached on restart. This over-counts on failure, which
            # fails safe.
            self.store.record_spend(self.inner.name, cost)

        result = self.inner.transcribe(segment)

        if self.mode == "record":
            path = self._path(segment)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(result, fh, sort_keys=True, indent=2)

        return result
