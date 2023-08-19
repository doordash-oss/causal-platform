include _infra/infra*.mk

.PHONY: docker-build
docker-build guard-ARTIFACTORY_USERNAME guard-ARTIFACTORY_PASSWORD guard-PIP_EXTRA_INDEX_URL:
	echo "${ARTIFACTORY_PASSWORD}" | docker login --username ${ARTIFACTORY_USERNAME} --password-stdin ddartifacts-docker.jfrog.io
	docker build -t causal-platform-builder \
		--build-arg ARTIFACTORY_USERNAME="$(subst @,%40,$(ARTIFACTORY_USERNAME))" \
		--build-arg ARTIFACTORY_PASSWORD="$(subst {,%7B,$(ARTIFACTORY_PASSWORD))" \
		--build-arg PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL}" \
		.


.PHONY: artifactory-upload
artifactory-upload: docker-build
	docker run \
		causal-platform-builder poetry publish -n --build -r "artifactory" -u "${ARTIFACTORY_USERNAME}" -p "${ARTIFACTORY_PASSWORD}"

clean:
	find . -name \*.pyc -delete
	find . -name \*.pyo -delete
	find . -name \*.cache -exec rm -rf {} +
	find . -name \__pycache__ -exec rm -rf {} +
	find . -name \*.pytest_cache -exec rm -rf {} +
	rm -Rf python/build
	rm -Rf python/dist
	rm -Rf python/*.egg-info

install-deps:
	pip install poetry==1.2.0
	pip install --upgrade pip
	poetry install --no-root
	pre-commit install
	pre-commit install-hooks

local-build:
	poetry build

unittest: clean
	pytest --cov-fail-under 85

lint:
	poetry run pre-commit run
