# simple makefile to simplify repetitive build env management tasks under posix

PYTHON ?= python
PYTEST ?= pytest

.PHONY: build develop install doc test xtest clean


build:
	$(PYTHON) setup.py build

develop: build
	$(PYTHON) setup.py develop

install: build
	$(PYTHON) setup.py install

doc: build
	$(PYTHON) setup.py build_sphinx

test: build
	$(PYTEST)

xtest: build
	$(PYTEST) -x

clean:
	$(PYTHON) setup.py clean
	find . -name __pycache__ -exec rm -rf {} +
	rm -rf .eggs *.egg-info
	rm -rf .coverage* .pytest_cache htmlcov
	rm -rf build doc/build dist
