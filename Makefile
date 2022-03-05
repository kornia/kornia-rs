.PHONY: all build test

all:
	make build

build:
	pip3 install -e .[dev]

test:
	pytest test/