#!/bin/bash
# Deployment script for Quick, Draw! Classification Training on Nautilus
# This script helps automate deployment to the NRP Nautilus cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load configuration from .env file
ENV_FILE="$(dirname "$0")/.env"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Quick, Draw! Classification Training${NC}"
echo -e "${BLUE}NRP Nautilus Deployment${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: .env file not found at ${ENV_FILE}${NC}"
    echo -e "${YELLOW}Please create a .env file from .env.template:${NC}"
    echo -e "  cp kubernetes/.env.template kubernetes/.env"
    echo -e "  # Edit kubernetes/.env with your values"
    exit 1
fi

# Load environment variables from .env
echo -e "${YELLOW}Loading configuration from .env...${NC}"
set -a  # automatically export all variables
source "$ENV_FILE"
set +a

# Validate required variables
REQUIRED_VARS=("NAMESPACE" "GITLAB_REGISTRY" "IMAGE_NAME" "IMAGE_TAG" "JOB_NAME")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${RED}Error: Missing required variables in .env:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo -e "  - ${var}"
    done
    echo -e "${YELLOW}Please check .env.template for required format${NC}"
    exit 1
fi

# Display loaded configuration
echo -e "${GREEN}✓ Configuration loaded:${NC}"
echo -e "  Namespace: ${BLUE}${NAMESPACE}${NC}"
echo -e "  Registry: ${BLUE}${GITLAB_REGISTRY}${NC}"
echo -e "  Image: ${BLUE}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "  Job Name: ${BLUE}${JOB_NAME}${NC}"

FULL_IMAGE="${GITLAB_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
PVC_NAME="${JOB_NAME}-pvc"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command_exists kubectl; then
    echo -e "${RED}Error: kubectl is not installed${NC}"
    exit 1
fi

if ! command_exists docker; then
    echo -e "${RED}Error: docker is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"

# Menu
echo -e "\n${YELLOW}What would you like to do?${NC}"
echo "1) Build and push Docker image"
echo "2) Create persistent volume claim"
echo "3) Submit training job to cluster"
echo "4) Check job status"
echo "5) View training logs (live or historical)"
echo "6) Download results from cluster"
echo "7) Delete job (cleanup)"
echo "8) Full deployment (steps 1-3)"
echo "9) View current configuration"
echo "10) Exit"
echo -n "Enter choice [1-10]: "
read choice

# Function to apply kubernetes yaml with variable substitution
apply_yaml_with_substitution() {
    local yaml_file=$1
    local temp_file="${yaml_file}.tmp"
    
    # Replace placeholders with actual values
    sed -e "s|NAMESPACE_PLACEHOLDER|${NAMESPACE}|g" \
        -e "s|IMAGE_PLACEHOLDER|${FULL_IMAGE}|g" \
        -e "s|quickdraw-training-job|${JOB_NAME}|g" \
        -e "s|quickdraw-results-pvc|${PVC_NAME}|g" \
        "${yaml_file}" > "${temp_file}"
    
    kubectl apply -f "${temp_file}"
    rm "${temp_file}"
}

case $choice in
    1)
        echo -e "\n${YELLOW}Building Docker image...${NC}"
        docker build -t ${FULL_IMAGE} -f kubernetes/Dockerfile .
        echo -e "${GREEN}✓ Image built${NC}"
        
        echo -e "${YELLOW}Pushing to registry...${NC}"
        docker push ${FULL_IMAGE}
        echo -e "${GREEN}✓ Image pushed to ${FULL_IMAGE}${NC}"
        ;;
    
    2)
        echo -e "\n${YELLOW}Creating persistent volume claim...${NC}"
        apply_yaml_with_substitution kubernetes/k8s-pvc.yaml
        echo -e "${GREEN}✓ PVC created${NC}"
        
        echo -e "${YELLOW}Waiting for PVC to be bound...${NC}"
        kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/${PVC_NAME} -n ${NAMESPACE} --timeout=120s
        echo -e "${GREEN}✓ PVC is bound and ready${NC}"
        ;;
    
    3)
        echo -e "\n${YELLOW}Submitting training job to cluster...${NC}"
        apply_yaml_with_substitution kubernetes/k8s-job.yaml
        echo -e "${GREEN}✓ Job submitted${NC}"
        
        echo -e "${YELLOW}Waiting for pod to be created...${NC}"
        sleep 5
        
        POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=quickdraw-training -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        if [ -n "$POD_NAME" ]; then
            echo -e "${GREEN}✓ Pod created: ${POD_NAME}${NC}"
            echo -e "${YELLOW}Checking pod status...${NC}"
            kubectl get pod ${POD_NAME} -n ${NAMESPACE}
        else
            echo -e "${RED}Warning: Pod not found yet. Check status with option 4.${NC}"
        fi
        ;;
    
    4)
        echo -e "\n${YELLOW}Checking job status...${NC}"
        echo -e "\n${BLUE}Job:${NC}"
        kubectl get job ${JOB_NAME} -n ${NAMESPACE} 2>/dev/null || echo "Job not found"
        
        echo -e "\n${BLUE}Pods:${NC}"
        kubectl get pods -n ${NAMESPACE} -l app=quickdraw-training
        
        POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=quickdraw-training -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        if [ -n "$POD_NAME" ]; then
            echo -e "\n${BLUE}Pod Details:${NC}"
            kubectl describe pod ${POD_NAME} -n ${NAMESPACE} | grep -A 10 "Conditions:\|Events:" | tail -15
            
            echo -e "\n${BLUE}GPU Allocation:${NC}"
            kubectl describe pod ${POD_NAME} -n ${NAMESPACE} | grep -A 2 "Limits:\|Requests:"
        fi
        ;;
    
    5)
        echo -e "\n${YELLOW}Fetching training logs...${NC}"
        POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=quickdraw-training -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        
        if [ -z "$POD_NAME" ]; then
            echo -e "${RED}No pod found for this job${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}Viewing logs from ${POD_NAME}...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}\n"
        
        echo "View options:"
        echo "1) Stream live logs (follow)"
        echo "2) View last 50 lines"
        echo "3) View last 200 lines"
        echo "4) View all logs"
        echo -n "Enter choice [1-4]: "
        read log_choice
        
        case $log_choice in
            1)
                kubectl logs -f ${POD_NAME} -n ${NAMESPACE}
                ;;
            2)
                kubectl logs ${POD_NAME} -n ${NAMESPACE} --tail=50
                ;;
            3)
                kubectl logs ${POD_NAME} -n ${NAMESPACE} --tail=200
                ;;
            4)
                kubectl logs ${POD_NAME} -n ${NAMESPACE}
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                exit 1
                ;;
        esac
        ;;
    
    6)
        echo -e "\n${YELLOW}Downloading results from cluster...${NC}"
        
        # Check if PVC exists
        echo -e "${YELLOW}Checking for PVC ${PVC_NAME} in namespace ${NAMESPACE}...${NC}"
        if ! kubectl get pvc ${PVC_NAME} -n ${NAMESPACE} 2>/dev/null; then
            echo -e "${RED}PVC ${PVC_NAME} not found. No results to download.${NC}"
            echo -e "${YELLOW}Available PVCs in namespace:${NC}"
            kubectl get pvc -n ${NAMESPACE}
            exit 1
        fi
        echo -e "${GREEN}✓ PVC found${NC}"
        
        # Try to download from running pod first
        echo -e "${YELLOW}Looking for running training pod...${NC}"
        POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=quickdraw-training --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
        
        DOWNLOADED=false
        
        if [ -n "$POD_NAME" ]; then
            echo -e "${GREEN}Found running pod: ${POD_NAME}${NC}"
            mkdir -p ./results_from_cluster
            echo -e "${YELLOW}Downloading from ${POD_NAME}:/workspace/runs/ ...${NC}"
            if kubectl cp ${NAMESPACE}/${POD_NAME}:/workspace/runs/ ./results_from_cluster/ 2>/dev/null; then
                DOWNLOADED=true
                echo -e "${GREEN}✓ Successfully downloaded from running pod${NC}"
            else
                echo -e "${RED}Failed to copy from running pod. Trying accessor pod method...${NC}"
            fi
        else
            echo -e "${YELLOW}No running pod found. Creating accessor pod...${NC}"
        fi
        
        if [ "$DOWNLOADED" = false ]; then
            # Generate unique pod name
            ACCESSOR_NAME="results-access-${RANDOM}"
            
            # Create accessor pod with inline YAML
            echo -e "${YELLOW}Creating accessor pod: ${ACCESSOR_NAME}${NC}"
            cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${ACCESSOR_NAME}
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  containers:
  - name: accessor
    image: busybox
    command: ["sleep", "3600"]
    volumeMounts:
    - name: results
      mountPath: /results
    resources:
      requests:
        memory: "128Mi"
        cpu: "100m"
      limits:
        memory: "256Mi"
        cpu: "200m"
  volumes:
  - name: results
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF
            
            # Wait with better error handling
            echo -e "${YELLOW}Waiting for accessor pod to be ready (timeout: 600s)...${NC}"
            if ! kubectl wait --for=condition=Ready pod/${ACCESSOR_NAME} -n ${NAMESPACE} --timeout=600s 2>/dev/null; then
                echo -e "${RED}Pod not ready. Checking status...${NC}"
                kubectl get pod ${ACCESSOR_NAME} -n ${NAMESPACE}
                echo -e "${YELLOW}Pod events:${NC}"
                kubectl describe pod ${ACCESSOR_NAME} -n ${NAMESPACE} | grep -A 10 "Events:"
                echo -e "${YELLOW}Cleaning up failed accessor pod...${NC}"
                kubectl delete pod ${ACCESSOR_NAME} -n ${NAMESPACE} 2>/dev/null || true
                echo -e "${RED}Failed to download results${NC}"
                exit 1
            fi
            
            mkdir -p ./results_from_cluster
            echo -e "${YELLOW}Downloading from PVC via ${ACCESSOR_NAME}:/results/ ...${NC}"
            kubectl cp ${NAMESPACE}/${ACCESSOR_NAME}:/results/ ./results_from_cluster/ 2>/dev/null || echo -e "${YELLOW}Note: /results/ may be empty or not exist${NC}"
            
            echo -e "${YELLOW}Cleaning up accessor pod...${NC}"
            kubectl delete pod ${ACCESSOR_NAME} -n ${NAMESPACE}
            DOWNLOADED=true
            echo -e "${GREEN}✓ Successfully downloaded from PVC${NC}"
        fi
        
        if [ "$DOWNLOADED" = true ] || [ -d ./results_from_cluster ]; then
            echo -e "\n${GREEN}✓ Results downloaded to ./results_from_cluster/${NC}"
            
            # Show what was downloaded
            echo -e "\n${BLUE}Downloaded files:${NC}"
            ls -lh ./results_from_cluster/
        else
            echo -e "${RED}Failed to download results${NC}"
        fi
        ;;
    
    7)
        echo -e "\n${YELLOW}Cleaning up job...${NC}"
        echo -e "${RED}This will delete the job and associated pods.${NC}"
        echo -n "Are you sure? (y/n): "
        read confirm
        
        if [ "$confirm" = "y" ]; then
            echo -e "${YELLOW}Deleting job...${NC}"
            kubectl delete job ${JOB_NAME} -n ${NAMESPACE} 2>/dev/null || echo "Job not found"
            echo -e "${GREEN}✓ Job deleted${NC}"
            
            echo -e "${YELLOW}Note: PVC (results storage) is preserved.${NC}"
            echo -e "${YELLOW}To delete PVC: kubectl delete pvc ${PVC_NAME} -n ${NAMESPACE}${NC}"
        else
            echo -e "${BLUE}Cleanup cancelled${NC}"
        fi
        ;;
    
    8)
        echo -e "\n${YELLOW}Running full deployment...${NC}"
        
        # Step 1: Build and push
        echo -e "\n${BLUE}[1/3] Building and pushing Docker image...${NC}"
        docker build -t ${FULL_IMAGE} -f kubernetes/Dockerfile .
        docker push ${FULL_IMAGE}
        echo -e "${GREEN}✓ Image ready${NC}"
        
        # Step 2: Create PVC
        echo -e "\n${BLUE}[2/3] Creating persistent volume claim...${NC}"
        apply_yaml_with_substitution kubernetes/k8s-pvc.yaml
        kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/${PVC_NAME} -n ${NAMESPACE} --timeout=120s
        echo -e "${GREEN}✓ PVC ready${NC}"
        
        # Step 3: Submit job
        echo -e "\n${BLUE}[3/3] Submitting training job...${NC}"
        apply_yaml_with_substitution kubernetes/k8s-job.yaml
        echo -e "${GREEN}✓ Job submitted${NC}"
        
        sleep 5
        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}Deployment Complete!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo -e "\n${YELLOW}Next steps:${NC}"
        echo -e "  • Monitor status: bash kubernetes/deploy.sh (option 4)"
        echo -e "  • View logs: bash kubernetes/deploy.sh (option 5)"
        echo -e "  • Download results: bash kubernetes/deploy.sh (option 6)"
        
        echo -e "\n${BLUE}Current status:${NC}"
        kubectl get pods -n ${NAMESPACE} -l app=quickdraw-training
        ;;
    
    9)
        echo -e "\n${BLUE}Current Configuration:${NC}"
        echo -e "  Namespace: ${GREEN}${NAMESPACE}${NC}"
        echo -e "  Registry: ${GREEN}${GITLAB_REGISTRY}${NC}"
        echo -e "  Image: ${GREEN}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
        echo -e "  Full Image: ${GREEN}${FULL_IMAGE}${NC}"
        echo -e "  Job Name: ${GREEN}${JOB_NAME}${NC}"
        echo -e "  PVC Name: ${GREEN}${PVC_NAME}${NC}"
        echo -e "\n${YELLOW}Configuration loaded from: ${ENV_FILE}${NC}"
        ;;
    
    10)
        echo -e "${BLUE}Exiting...${NC}"
        exit 0
        ;;
    
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Done!${NC}"
