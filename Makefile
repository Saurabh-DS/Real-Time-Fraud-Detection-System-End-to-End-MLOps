# Makefile for Real-Time Fraud Detection System
# ==============================================
# Commands for development, building, and deployment.

.PHONY: help dev up down train test build push deploy logs clean

# Default target
help:
	@echo "================================================"
	@echo "  Real-Time Fraud Detection System (Helm)"
	@echo "================================================"
	@echo ""
	@echo "Development Commands:"
	@echo "  make dev          - Start Docker Compose stack"
	@echo "  make dev-down     - Stop Docker Compose stack"
	@echo "  make port-forward - Start ALL port forwarding scripts"
	@echo ""
	@echo "Kubernetes Commands (Helm):"
	@echo "  make deploy       - Install/Upgrade Helm chart"
	@echo "  make delete       - Uninstall Helm chart"
	@echo "  make status       - Check pod status"
	@echo ""
	@echo "Build Commands:"
	@echo "  make build        - Build all Docker images"
	@echo "  make push         - Push images to registry"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make logs         - Tail logs from all pods"
	@echo "  make clean        - Clean up Docker resources"
	@echo ""

# ==========================================
# Development Commands (Docker Compose)
# ==========================================

# Start local development stack
# Standard Docker Compose iteration

dev:
	@echo "Starting Docker Compose development stack..."

	docker-compose up -d
	@echo "Services starting. Access points:"

	@echo "   - Fraud API:    http://localhost:8000"
	@echo "   - MLflow:       http://localhost:5000"
	@echo "   - Grafana:      http://localhost:3000"
	@echo "   - Airflow:      http://localhost:8080"
	@echo "   - Prometheus:   http://localhost:9090"

# Stop development stack
dev-down:
	@echo "Stopping Docker Compose stack..."

	docker-compose down
	@echo "All services stopped"


# Train model with GPU acceleration
# Leverage available GPU for training

train:
	@echo "Training fraud detection model with GPU..."

	@echo "   GPU Device: NVIDIA RTX 4060 (8GB)"
	cd training && python train.py --gpu
	@echo "Training complete. Check MLflow for results."


# Run tests
test:
	@echo "Running test suite..."
	pytest tests/ -v --cov=serving --cov=features
	@echo "Tests complete"

# ==========================================
# Kubernetes Commands (Kind + Helm)
# ==========================================

# Create Kind cluster and deploy everything
# Kubernetes environment setup

up: cluster-create cluster-setup deploy
	@echo ""
	@echo "================================================"
	@echo "Fraud Detection System deployed to Kind!"
	@echo "================================================"
	@echo ""
	@echo "Access points (via NodePort):"
	@echo "   - Fraud API:    http://localhost:30000"
	@echo "   - Grafana:      http://localhost:30001"
	@echo "   - MLflow:       http://localhost:30002"

# Create Kind cluster
cluster-create:
	@echo "Creating Kind cluster..."
	kind create cluster --config kind-config.yaml --image kindest/node:v1.31.1 --wait 60s
	@echo "Cluster created"

# Setup cluster prerequisites
cluster-setup:
	@echo "Installing cluster prerequisites..."
	# Create namespaces
	kubectl apply -f k8s/namespace.yaml
	# Install NGINX Ingress Controller
	kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
	# Wait for ingress to be ready
	kubectl wait --namespace ingress-nginx \
		--for=condition=ready pod \
		--selector=app.kubernetes.io/component=controller \
		--timeout=120s
	@echo "Cluster setup complete"

# ==========================================
# Kubernetes Commands (Helm)
# ==========================================

# Deploy using Helm
deploy:
	@echo "Deploying fraud detection services via Helm..."
	helm upgrade --install fraud-stack ./helm/fraud-detection-system --wait --timeout 600s
	@echo "Deployment complete"

# Uninstall Helm release
delete:
	@echo "Uninstalling Helm release..."
	helm uninstall fraud-stack
	@echo "Release uninstalled"

# Start Port Forwarding (Automatic detection)
port-forward:
	@echo "Starting port forwarding..."
ifeq ($(OS),Windows_NT)
	powershell -ExecutionPolicy Bypass -File ./scripts/start-port-forwarding.ps1
else
	./scripts/start-port-forwarding.sh
endif

# Delete Kind cluster
down:
	@echo "Deleting Kind cluster..."
	kind delete cluster --name fraud-detection
	@echo "Cluster deleted"

# ==========================================
# Build Commands
# ==========================================

# Build all Docker images
build:
	@echo "Building Docker images..."
	docker-compose build
	@echo "Images built"

# Build and load images to Kind
build-kind: build
	@echo "Loading images to Kind cluster..."
	kind load docker-image fraud-serving:latest --name fraud-detection
	kind load docker-image fraud-producer:latest --name fraud-detection
	kind load docker-image feature-processor:latest --name fraud-detection
	@echo "Images loaded to Kind"

# ==========================================
# Utility Commands
# ==========================================

# Tail logs from fraud-api pods
logs:
	kubectl logs -f -l app=fraud-api --namespace fraud-detection --all-containers

# Port forward fraud-api for local access
port-forward:
	kubectl port-forward svc/fraud-api 8000:8000 --namespace fraud-detection

# Get cluster status
status:
	@echo "Cluster Status:"
	@echo ""
	@echo "Nodes:"
	kubectl get nodes -o wide
	@echo ""
	@echo "Pods:"
	kubectl get pods --all-namespaces
	@echo ""
	@echo "Services:"
	kubectl get svc --all-namespaces

# Clean Docker resources
clean:
	@echo "Cleaning up Docker resources..."

	docker system prune -f
	docker volume prune -f
	@echo "Cleanup complete"


# Full reset (cluster + Docker)
reset: down clean
	@echo "Full reset complete"
