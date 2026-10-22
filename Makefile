PYTHON ?= python

.PHONY: data graphs train eval figures test

data:
	$(PYTHON) scripts/01_audit_and_download.py
	$(PYTHON) scripts/02_preprocess.py
	$(PYTHON) scripts/03_segment.py

graphs:
	$(PYTHON) scripts/04_build_graphs.py

train:
	$(PYTHON) scripts/06_train.py

eval:
	$(PYTHON) scripts/07_evaluate.py

figures:
	$(PYTHON) scripts/09_make_figures.py

test:
	$(PYTHON) -m pytest
