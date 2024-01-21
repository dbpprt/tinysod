all: build

build: pre-commit
	mypy --config-file=pyproject.toml .

test: build
	# test commands here

pre-commit:
	pre-commit run --all-files
