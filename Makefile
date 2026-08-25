.PHONY: test bench track clean-track

# Override on the command line, e.g. `make bench AUDIO=clips/sample.wav`.
AUDIO       ?= samples/fleurs-hi-12091698556182716328.wav
TRACK       ?= results/track.json
DELTA_TABLE ?= delta_table.json
DHVANI_MODE ?= replay

test:
	uv run pytest -v

# Fix round 2 (I4): bench used to read results/track.json, which nothing in
# the project ever wrote, so the target could not run. The CLI now persists
# the track via --out, and bench builds it first.
#
# A clean clone CAN now run this: the demo audio and its Tier 0 replay
# fixtures are both committed. `delta_table.json` is not -- it is the output
# of `dhvani-calibrate`, which needs the real corpus -- so bench renders a
# frontier of zeros until a calibration run produces one. That is honest
# rather than broken: with no measured deltas, nothing is worth escalating.
# The default AUDIO is real speech committed under samples/ (see
# samples/ATTRIBUTION.md): a clean clone already has it, and its replay
# fixtures are committed too, so `make track` needs no model and no download.
#
# sample.wav is the synthetic sweep scripts/spike_chirp.py uses. It stays
# gitignored and is generated on demand; its fixtures are committed as well,
# and the generator is deterministic, so the filenames still match.
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
