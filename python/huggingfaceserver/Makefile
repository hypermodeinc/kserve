IMG ?= hypermode/huggingface-model-server:latest

dev_install:
	poetry install --with test -E vllm --no-interaction

install_dependencies:
	poetry install --with=test -E vllm --no-interaction --no-root
	
test: type_check
	pytest -W ignore

type_check:
	mypy --ignore-missing-imports huggingfaceserver 

build-docker:
	docker build --platform linux/amd64 -t $(IMG) -f ../huggingface_server.Dockerfile ..
