.DEFAULT_GOAL := install

.PHONY: lock install-nocuda install fmt test clean docker-build docker-clean

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

lock:
	docker-compose run --rm lock

install:
	${ROOT_DIR}/scripts/install.sh

install-nocuda:
	${ROOT_DIR}/scripts/install.sh requirements-nocuda.txt

build:
	docker build \
		--tag deep_learning_at_scale \
		--file Dockerfile .

check: build
	docker-compose run -v "/tmp/mypy-cache:/app/.mypy_cache" \
		--rm deep-learning-at-scale scripts/lint.sh

test: build
	docker-compose run --rm test ; \
		$(MAKE) -s down status=$$?

down:
	docker-compose down
	exit $(status)

fmt:
	${ROOT_DIR}/scripts/lint.sh -f fixup

clean:
	rm -rf deep_learning_at_scale.egg-info .coverage .mypy_cache .pytest_cache
	find . -type f -name '*.pyc' -delete
	find . -type d -name __pycache__ -delete

docker-build:
	DOCKER_BUILDKIT=1 docker build \
		--tag deep_learning_at_scale .

docker-clean:
	docker-compose down
	exit $(status)
