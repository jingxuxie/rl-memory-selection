PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: install test population finite-sample scaling weighted random-stress \
	experiments figures paper reproduce submission-check package clean

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m compileall -q src experiments tests
	$(PYTHON) -m pytest -q

population:
	$(PYTHON) experiments/run_population.py

finite-sample:
	$(PYTHON) experiments/run_finite_sample.py

scaling:
	$(PYTHON) experiments/run_scaling.py

weighted:
	$(PYTHON) experiments/run_weighted.py

random-stress:
	$(PYTHON) experiments/run_random_stress.py

experiments: population finite-sample scaling weighted random-stress

figures:
	$(PYTHON) experiments/make_figures.py

paper:
	bash scripts/build_paper.sh

submission-check: paper
	bash scripts/submission_check.sh

package: submission-check
	bash scripts/package_submission.sh

reproduce: test experiments figures paper submission-check

clean:
	rm -rf .pytest_cache rendered_main rendered_supp submission
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.fdb_latexmk \
		paper/*.fls paper/*.log paper/*.out paper/*.synctex.gz paper/*.toc
