PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
MODEL ?= svm
TASK ?= three_class
FOLD ?=
RUN ?=

.PHONY: p1 p2 p3 p4 p5 data graphs train train-all evaluate evaluate-all eval benchmark plots smoke test

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

train-all:
	PYTHONPATH=src $(PYTHON) -m chronograph.training.all_folds --model $(MODEL) --task $(TASK)

test:
	$(PYTHON) -m pytest


evaluate:
	test -n "$(FOLD)" -a -n "$(RUN)"
	PYTHONPATH=src $(PYTHON) -m chronograph.evaluation.evaluate --model $(MODEL) --task $(TASK) --fold $(FOLD) --run $(RUN)

evaluate-all:
	PYTHONPATH=src $(PYTHON) -m chronograph.evaluation.all_folds --model $(MODEL) --task $(TASK)

eval:
	PYTHONPATH=src $(PYTHON) -m chronograph.evaluation.aggregate

benchmark:
	PYTHONPATH=src $(PYTHON) -m chronograph.models.slate.benchmark

smoke:
	PYTHONPATH=src $(PYTHON) -m chronograph.smoke

plots:
	PYTHONPATH=src $(PYTHON) -m chronograph.evaluation.plots
