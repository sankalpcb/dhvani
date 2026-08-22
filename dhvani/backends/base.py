"""Backend protocol plus the record/replay wrapper.

Three rules, all fail-closed:
  1. Replay never falls back to live on a cache miss.
  2. Live calls atomically reserve the spend — check the ledger and record
     it in one transaction — BEFORE the paid call executes (not after).
     See RULING and FIX ROUND 2 below.
  3. Live and record modes both make paid calls to the inner backend, so
     both require a Store — constructing either with store=None raises
     ValueError instead of silently running unmetered. Only replay may omit
     a store, since replay never calls anything.

RULING (overrides an earlier draft of this module): the live path records
spend before invoking the inner backend, not after. If the spend were
recorded after inner.transcribe(), a crash between the API call and the
record would leave money spent but unrecorded, so total_spend() would
under-count and the USD 20 ceiling could be breached on restart. Recording
first is pessimistic accounting — a failed call over-counts, which fails
safe. Preserving the ceiling matters more than perfectly accurate cost
attribution.

FIX ROUND 2 (C1): the ordering above was right but insufficient. This
module called store.check_budget() and store.record_spend() as two
separate statements, each in its own autocommitted transaction, so two
Recorded wrappers holding two Store handles on the same DB file could both
read the same stale total, both pass the check, and both then record —
ending above the ceiling with neither raising. dhvani.cli's --db defaults
to a fixed shared path, so this is reachable from the shipped CLI. The
call is now the single atomic store.reserve_spend(), which does the check
and the insert in ONE statement. The before-the-call ordering above is
preserved: the reservation completes before inner.transcribe() runs.

FIX ROUND 2 (I2/I3): fixtures were keyed on {segment_id}.json under the
inner backend's name alone, and segment_id hashes PCM only — so recording
with lang="hi" and replaying with lang="ml" silently returned the Hindi
response, and bumping POLICY_ID invalidated nothing. Fixture paths now
carry ids.variant_slug(variant_key), which folds in POLICY_ID. A stale
variant is a missing fixture, and a missing fixture is still a hard error:
replay never falls back to live.

FIX ROUND 1: a review found that Recorded(inner, mode="live", store=None)
constructed fine and then skipped both check_budget() and record_spend()
entirely, calling the paid backend with zero enforcement and zero
accounting — the USD 20 ceiling was simply absent. record mode has the same
hole (it also calls the live backend, then saves the response). __init__
now rejects store=None for both live and record modes.
"""

import json
import os
from typing import Literal, Protocol, runtime_checkable

from dhvani.ids import variant_slug

Mode = Literal["record", "replay", "live"]


class FixtureMissing(RuntimeError):
    """Replay mode was asked for a segment with no recorded fixture."""


@runtime_checkable
class Backend(Protocol):
    name: str

    variant_key: str
    """Short, stable string capturing everything about this backend's
    configuration that changes the output for byte-identical PCM (lang,
    model_id, recognizer). See FIX ROUND 2 (I2/I3) in the module
    docstring. Required, not optional: a backend that omits it would
    silently share cache entries and fixtures with every other
    configuration of the same tier."""

    def cost_per_call(self, segment) -> float: ...

    def transcribe(self, segment) -> dict: ...


class Recorded:
    """Wraps a Backend with record/replay and budget enforcement."""

    def __init__(self, inner: Backend, mode: Mode, fixture_dir: str, store=None):
        if mode not in ("record", "replay", "live"):
            raise ValueError(f"unknown mode: {mode}")
        if mode in ("live", "record") and store is None:
            raise ValueError(
                f"mode={mode!r} makes paid calls to the inner backend and "
                f"requires a Store to enforce the spend ceiling; got store=None. "
                f"Only mode='replay' may omit a store, since replay never calls "
                f"anything."
            )
        self.inner = inner
        self.mode = mode
        self.fixture_dir = fixture_dir
        self.store = store
        self.name = inner.name
        # Read eagerly so a backend missing the required protocol member
        # fails loudly at construction, not silently at cache-lookup time.
        self.variant_key = inner.variant_key

    def _path(self, segment) -> str:
        # The variant slug is a path component, so two configurations of
        # the same tier (different lang, model_id, recognizer) and two
        # different POLICY_IDs cannot collide on one fixture file.
        return os.path.join(
            self.fixture_dir,
            self.inner.name,
            variant_slug(self.variant_key),
            f"{segment.segment_id}.json",
        )

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
            # ONE atomic statement that both checks the ceiling and records
            # the spend, and it runs BEFORE the paid call — see the module
            # docstring for both rulings. Splitting this back into
            # check_budget() + record_spend() reopens the C1 race.
            self.store.reserve_spend(self.inner.name, cost)

        result = self.inner.transcribe(segment)

        if self.mode == "record":
            path = self._path(segment)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(result, fh, sort_keys=True, indent=2)

        return result
