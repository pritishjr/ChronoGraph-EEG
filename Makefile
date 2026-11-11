PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: p1 p2 p3 p4 p5 data graphs train test

p1:
	PYTHONPATH=src $(PYTHON) -m chronograph.data.channel_audit

p2:
	PYTHONPATH=src $(PYTHON) -m chronograph.data.preprocessing

p3:
	PYTHONPATH=src $(PYTHON) -m chronograph.data.segmentation

p4:
	PYTHONPATH=src $(PYTHON) -m chronograph.graphs.graph_builder

p5:
	PYTHONPATH=src $(PYTHON) -m chronograph.splits

data:
	$(MAKE) p1
	$(MAKE) p3
	$(MAKE) p2

graphs:
	$(MAKE) p4

train:
	PYTHONPATH=src $(PYTHON) -m chronograph.training --help

test:
	$(PYTHON) -m pytest
