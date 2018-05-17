# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows

PYTHON ?= python
PYTESTS ?= py.test
CTAGS ?= ctags
CODESPELL_SKIPS ?= "*.fif"
CODESPELL_DIRS ?= mne_g3d/
all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf _build

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test:
	pytest mne_g3d

pep: flake pydocstyle codespell-error

flake:
	flake8 --count mne_g3d

pydocstyle:
	pydocstyle

codespell-error:  # running on travis
	codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) $(CODESPELL_DIRS)
