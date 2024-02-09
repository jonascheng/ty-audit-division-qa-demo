.DEFAULT_GOAL := help

APPLICATION?=taoyuan-audit-division-qa-demo
COMMIT_SHA?=$(shell git rev-parse --short HEAD)
DOCKER?=docker
DOCKERHUB_OWNER?=jonascheng
DOCKER_IMG_NAME=${DOCKERHUB_OWNER}/${APPLICATION}
PWD?=$(shell pwd)

.PHONY: setup
setup: ## setup
	python -m pip install --upgrade pip
	pip install -r requirements.txt -q

.PHONY: law-n-order-load-transform
law-n-order-load-transform: setup ## load and transform data for embeddings
	python app/main.py --transform-law-n-order

.PHONY: run
run: setup ## run
	streamlit run app/app.py

.PHONY: law-embeddings
law-embeddings: setup ## create law embeddings
	rm -rf assets/law.json/ChLaw_embeddings.chorma
	python app/main.py --transform-law-embeddings

.PHONY: order-embeddings
order-embeddings: setup ## create order embeddings
	rm -rf assets/order.json/ChOrder_embeddings.chorma
	python app/main.py --transform-order-embeddings

.PHONY: docker-build
docker-build: ## build docker image
	${DOCKER} build -t ${DOCKER_IMG_NAME}:${COMMIT_SHA} .

.PHONY: docker-push
docker-push: docker-build ## push docker image
	${DOCKER} push ${DOCKER_IMG_NAME}:${COMMIT_SHA}

.PHONY: help
help: ## prints this help message
	@echo "Usage: \n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
