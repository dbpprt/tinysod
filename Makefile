CONDA_ENV := tinysod

all: activate build

build: activate pre-commit
	mypy --config-file=pyproject.toml .

test: build
	# test commands here

pre-commit: activate
	pre-commit run --all-files

activate:
	# conda activate $(CONDA_ENV)
