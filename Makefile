.PHONY: test bench track clean-track

# Override on the command line, e.g. `make bench AUDIO=clips/sample.wav`.
AUDIO       ?= sample.wav
TRACK       ?= results/track.json
DELTA_TABLE ?= delta_table.json
DHVANI_MODE ?= replay

test:
	uv run pytest -v

# Fix round 2 (I4): bench used to read results/track.json, which nothing in
# the project ever wrote, so the target could not run. The CLI now persists
# the track via --out, and bench builds it first.
#
# `fixtures/` and `delta_table.json` are deferred to Phase 2 (see the plan's
# Phase 1 Exit Criteria), so a clean clone has no audio to transcribe and no
# measured deltas. bench is wired correctly and will fail loudly on the
# missing input rather than reporting on a file nobody produced.
# A clean clone has no sample.wav (*.wav is gitignored), so generate the one
# the committed replay fixtures were recorded from. Deterministic: the
# fixture filenames are hashes of exactly this signal.
sample.wav:
	uv run python scripts/make_sample_wav.py

$(TRACK): $(AUDIO)
	mkdir -p $(dir $(TRACK))
	DHVANI_MODE=$(DHVANI_MODE) uv run dhvani $(AUDIO) --out $(TRACK)

track: $(TRACK)

bench: $(TRACK)
	mkdir -p $(dir $(TRACK))
	uv run dhvani-bench $(TRACK) $(DELTA_TABLE) > results/report.md
	@echo "wrote results/report.md"

clean-track:
	rm -f $(TRACK)
