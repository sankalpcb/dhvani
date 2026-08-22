.PHONY: test bench

test:
	uv run pytest -v

bench:
	mkdir -p results
	DHVANI_MODE=replay uv run python -m dhvani.report_cli \
	  results/track.json delta_table.json > results/report.md
	@echo "wrote results/report.md"
