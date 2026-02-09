#!/bin/bash
# Kubernetes Auto-scaling Dashboard (Bash)
# Monitors HPA scale-up and scale-down driven by internal Producer load.

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

function write_step {
    echo -e "\n${CYAN}-> $1${NC}"
}

function write_success {
    echo -e "${GREEN}✓ $1${NC}"
}

MAX_REPLICAS=15

clear
write_step "Starting Dashboard (Producer Load: 1000 TPS)..."
echo "Press Ctrl+C to abort."

while true; do
    # 1. HPA Status
    REPLICAS=$(kubectl get hpa fraud-stack-fraud-detection-system-serving-hpa -n default -o jsonpath='{.status.currentReplicas}')
    CPU=$(kubectl get hpa fraud-stack-fraud-detection-system-serving-hpa -n default -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}')
    
    if [ -z "$REPLICAS" ]; then REPLICAS=0; fi
    if [ -z "$CPU" ]; then CPU=0; fi
    
    clear
    echo "---------------------------------------------------"
    echo "HPA Status: $REPLICAS / $MAX_REPLICAS Replicas | CPU Load: $CPU%"
    echo "---------------------------------------------------"
    
    # 2. Pod Metrics (RAM)
    echo -e "\nPod Resources:"
    kubectl top pods -n default -l app=fraud-stack-fraud-detection-system-serving --no-headers | sort
    
    # 3. Pod Status
    echo -e "\nPod Lifecycle Status:"
    kubectl get pods -n default -l app=fraud-stack-fraud-detection-system-serving -o wide --show-labels
    
    # 4. Cluster Topology (Node Distribution)
    echo -e "\nCluster Topology (Where are my containers?):"
    kubectl get pods -n default -l app=fraud-stack-fraud-detection-system-serving -o custom-columns=NODE:.spec.nodeName,NAME:.metadata.name,STATUS:.status.phase,CONTAINER_ID:.status.containerStatuses[0].containerID --sort-by=.spec.nodeName
    
    sleep 3
done
