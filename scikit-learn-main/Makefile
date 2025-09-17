# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= pytest
CTAGS ?= ctags

# skip doctests on 32bit python
BITS := $(shell python -c 'import struct; print(8 * struct.calcsize("P"))')

all: clean inplace test

clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

dev-meson:
	pip install --verbose --no-build-isolation --editable . --config-settings editable-verbose=true

clean-meson:
	pip uninstall -y custom-scikit-learn

test-code: in
	$(PYTEST) --showlocals -v custom_sklearn --durations=20
test-sphinxext:
	$(PYTEST) --showlocals -v doc/sphinxext/
test-doc:
ifeq ($(BITS),64)
	$(PYTEST) $(shell find doc -name '*.rst' | sort)
endif
test-code-parallel: in
	$(PYTEST) -n auto --showlocals -v custom_sklearn --durations=20

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) custom_sklearn --showlocals -v --cov=custom_sklearn --cov-report=html:coverage
test-coverage-parallel:
	rm -rf coverage .coverage .coverage.*
	$(PYTEST) custom_sklearn -n auto --showlocals -v --cov=custom_sklearn --cov-report=html:coverage

test: test-code test-sphinxext test-doc

trailing-spaces:
	find custom_sklearn -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

cython:
	python setup.py build_src

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R custom_sklearn

doc: inplace
	$(MAKE) -C doc html

doc-noplot: inplace
	$(MAKE) -C doc html-noplot

code-analysis:
	build_tools/linting.sh
