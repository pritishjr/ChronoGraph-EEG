PYTHON = python3

.PHONY: p1 data graphs train eval figures test

p1:
	PYTHONPATH=src $(PYTHON) -m chronograph.data.channel_audit

data:
	$(MAKE) p1

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
