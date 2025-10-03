# Makefile for Portalis NIM Microservices

.PHONY: help build test deploy clean

# Variables
IMAGE_NAME := portalis-nim
IMAGE_TAG := latest
NAMESPACE := portalis
HELM_RELEASE := portalis-nim

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Development
dev: ## Run development server
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	pytest tests/ -v

test-coverage: ## Run tests with coverage
	pytest tests/ --cov=api --cov-report=html --cov-report=term

lint: ## Run linting
	black api/ grpc/ config/
	ruff check api/ grpc/ config/

format: ## Format code
	black api/ grpc/ config/

# Docker
build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

build-optimized: ## Build optimized Docker image
	docker build -f Dockerfile.optimized -t $(IMAGE_NAME):optimized .

push: ## Push Docker image
	docker push $(IMAGE_NAME):$(IMAGE_TAG)

run-docker: ## Run with Docker Compose
	docker-compose up -d

stop-docker: ## Stop Docker Compose
	docker-compose down

logs-docker: ## View Docker Compose logs
	docker-compose logs -f

# Kubernetes
k8s-deploy: ## Deploy to Kubernetes with kubectl
	kubectl apply -f k8s/base/ --namespace $(NAMESPACE)

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -f k8s/base/ --namespace $(NAMESPACE)

k8s-status: ## Check Kubernetes deployment status
	kubectl get pods,svc,ing -n $(NAMESPACE) -l app=portalis-nim

k8s-logs: ## View Kubernetes logs
	kubectl logs -n $(NAMESPACE) -l app=portalis-nim --tail=100 -f

# Helm
helm-install: ## Install with Helm
	helm install $(HELM_RELEASE) ./helm --namespace $(NAMESPACE) --create-namespace

helm-upgrade: ## Upgrade Helm release
	helm upgrade $(HELM_RELEASE) ./helm --namespace $(NAMESPACE)

helm-uninstall: ## Uninstall Helm release
	helm uninstall $(HELM_RELEASE) --namespace $(NAMESPACE)

helm-template: ## Show Helm template output
	helm template $(HELM_RELEASE) ./helm

helm-lint: ## Lint Helm chart
	helm lint ./helm

# gRPC
grpc-generate: ## Generate gRPC code from proto files
	python -m grpc_tools.protoc \
		-I./grpc \
		--python_out=./grpc \
		--grpc_python_out=./grpc \
		./grpc/translation.proto

# Monitoring
port-forward-api: ## Port forward API service
	kubectl port-forward svc/$(HELM_RELEASE)-api 8000:8000 -n $(NAMESPACE)

port-forward-metrics: ## Port forward metrics
	kubectl port-forward svc/$(HELM_RELEASE)-api 9090:9090 -n $(NAMESPACE)

port-forward-prometheus: ## Port forward Prometheus
	kubectl port-forward svc/prometheus 9091:9090 -n monitoring

port-forward-grafana: ## Port forward Grafana
	kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Utilities
clean: ## Clean temporary files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage

docs: ## Build documentation
	cd docs && mkdocs build

docs-serve: ## Serve documentation locally
	cd docs && mkdocs serve

# CI/CD
ci-test: install lint test ## Run CI pipeline tests

ci-build: build ## Run CI pipeline build

# All-in-one commands
all: clean install lint test build ## Run all checks and build

deploy-local: build run-docker ## Build and deploy locally with Docker

deploy-k8s: build push helm-upgrade ## Build, push, and deploy to Kubernetes
