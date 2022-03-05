.PHONY: all build test

all:
	make build

build:
	pip3 install -e .[dev] --user

test:
	pytest test/