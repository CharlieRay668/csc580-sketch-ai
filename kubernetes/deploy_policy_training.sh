#!/bin/bash
# Deployment script for Guessing Policy Training on Nautilus
# Trains policies in multi-agent competitive setting

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
echo -e "${BLUE}Guessing Policy Training${NC}"
echo -e "${BLUE}NRP Nautilus Deployment${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: .env file not found at ${ENV_FILE}${NC}"
    echo -e "${YELLOW}Please create a .env file${NC}"
    exit 1
fi

# Load environment variables from .env
echo -e "${YELLOW}Loading configuration from .env...${NC}"
set -a
source "$ENV_FILE"
set +a

# Override image name and job name for policy training
IMAGE_NAME="quickdraw-policy-training"
JOB_NAME="quickdraw-policy-training"
PVC_NAME="quickdraw-policy-results-pvc"

# Validate required variables
REQUIRED_VARS=("NAMESPACE" "GITLAB_REGISTRY")
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
    exit 1
fi

# Display loaded configuration
echo -e "${GREEN}✓ Configuration loaded:${NC}"
echo -e "  Namespace: ${BLUE}${NAMESPACE}${NC}"
echo -e "  Registry: ${BLUE}${GITLAB_REGISTRY}${NC}"
echo -e "  Image: ${BLUE}${IMAGE_NAME}:latest${NC}"
echo -e "  Job Name: ${BLUE}${JOB_NAME}${NC}"

FULL_IMAGE="${GITLAB_REGISTRY}/${IMAGE_NAME}:latest"

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
echo "3) Submit policy training job to cluster"
echo "4) Check job status"
echo "5) View training logs (live)"
echo "6) Download trained policies from cluster"
echo "7) Delete job (cleanup)"
echo "8) Full deployment (steps 1-3)"
echo "9) Exit"
echo -n "Enter choice [1-9]: "
read choice

case $choice in
    1)
        echo -e "\n${YELLOW}Building and pushing Docker image...${NC}"
        echo -e "${BLUE}Building for linux/amd64 platform...${NC}"
        
        cd "$(dirname "$0")/.."
        
        # Build and push using buildx for correct architecture
        docker buildx build \
            --platform linux/amd64 \
            -f kubernetes/Dockerfile.policy-training \
            -t "${FULL_IMAGE}" \
            --push \
            .
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Image built and pushed successfully${NC}"
            echo -e "  Image: ${BLUE}${FULL_IMAGE}${NC}"
        else
            echo -e "${RED}✗ Failed to build/push image${NC}"
            exit 1
        fi
        ;;
        
    2)
        echo -e "\n${YELLOW}Creating persistent volume claim...${NC}"
        
        # Create temp file with substitutions
        TMP_PVC="/tmp/k8s-policy-pvc-$$.yaml"
        sed -e "s|NAMESPACE_PLACEHOLDER|${NAMESPACE}|g" \
            "$(dirname "$0")/k8s-policy-training-pvc.yaml" > "${TMP_PVC}"
        
        kubectl apply -f "${TMP_PVC}"
        rm "${TMP_PVC}"
        
        echo -e "${GREEN}✓ PVC created: ${PVC_NAME}${NC}"
        ;;
        
    3)
        echo -e "\n${YELLOW}Submitting policy training job...${NC}"
        
        # Check if PVC exists
        if ! kubectl get pvc "${PVC_NAME}" -n "${NAMESPACE}" &>/dev/null; then
            echo -e "${RED}Error: PVC ${PVC_NAME} does not exist${NC}"
            echo -e "${YELLOW}Run option 2 to create it first${NC}"
            exit 1
        fi
        
        # Create temp file with substitutions
        TMP_JOB="/tmp/k8s-policy-job-$$.yaml"
        sed -e "s|NAMESPACE_PLACEHOLDER|${NAMESPACE}|g" \
            -e "s|IMAGE_PLACEHOLDER|${FULL_IMAGE}|g" \
            "$(dirname "$0")/k8s-policy-training-job.yaml" > "${TMP_JOB}"
        
        kubectl apply -f "${TMP_JOB}"
        rm "${TMP_JOB}"
        
        echo -e "${GREEN}✓ Job submitted: ${JOB_NAME}${NC}"
        echo -e "${YELLOW}Monitor with: kubectl logs -f job/${JOB_NAME} -n ${NAMESPACE}${NC}"
        ;;
        
    4)
        echo -e "\n${YELLOW}Checking job status...${NC}"
        kubectl get jobs "${JOB_NAME}" -n "${NAMESPACE}" 2>/dev/null || echo -e "${RED}Job not found${NC}"
        echo ""
        kubectl get pods -l job-name="${JOB_NAME}" -n "${NAMESPACE}" 2>/dev/null || echo -e "${RED}No pods found${NC}"
        ;;
        
    5)
        echo -e "\n${YELLOW}Viewing training logs (live)...${NC}"
        kubectl logs -f "job/${JOB_NAME}" -n "${NAMESPACE}"
        ;;
        
    6)
        echo -e "\n${YELLOW}Downloading trained policies...${NC}"
        
        # Create accessor pod
        ACCESS_POD="access-policy-results"
        
        echo -e "${BLUE}Creating accessor pod...${NC}"
        cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${ACCESS_POD}
  namespace: ${NAMESPACE}
spec:
  containers:
  - name: accessor
    image: busybox
    command: ['sh', '-c', 'sleep 3600']
    volumeMounts:
    - name: results
      mountPath: /data
  volumes:
  - name: results
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF
        
        echo -e "${BLUE}Waiting for pod to be ready...${NC}"
        kubectl wait --for=condition=ready pod/${ACCESS_POD} -n ${NAMESPACE} --timeout=60s
        
        # List files
        echo -e "${BLUE}Files in PVC:${NC}"
        kubectl exec -n ${NAMESPACE} ${ACCESS_POD} -- ls -lh /data/
        
        # Download
        OUTPUT_DIR="$(dirname "$0")/trained_policies"
        mkdir -p "${OUTPUT_DIR}"
        
        echo -e "${BLUE}Downloading to ${OUTPUT_DIR}/${NC}"
        kubectl cp ${NAMESPACE}/${ACCESS_POD}:/data/ "${OUTPUT_DIR}/"
        
        echo -e "${GREEN}✓ Policies downloaded to ${OUTPUT_DIR}/${NC}"
        
        # Cleanup accessor pod
        echo -e "${BLUE}Cleaning up accessor pod...${NC}"
        kubectl delete pod ${ACCESS_POD} -n ${NAMESPACE}
        ;;
        
    7)
        echo -e "\n${YELLOW}Deleting job and cleaning up...${NC}"
        echo -e "${RED}Warning: This will delete the job but NOT the PVC${NC}"
        echo -n "Continue? [y/N]: "
        read confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}" 2>/dev/null || echo -e "${YELLOW}Job not found${NC}"
            echo -e "${GREEN}✓ Cleanup complete${NC}"
        fi
        ;;
        
    8)
        echo -e "\n${YELLOW}Full deployment...${NC}"
        
        # Step 1: Build
        echo -e "\n${BLUE}[1/3] Building and pushing Docker image...${NC}"
        cd "$(dirname "$0")/.."
        docker buildx build \
            --platform linux/amd64 \
            -f kubernetes/Dockerfile.policy-training \
            -t "${FULL_IMAGE}" \
            --push \
            .
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Build failed${NC}"
            exit 1
        fi
        
        # Step 2: PVC
        echo -e "\n${BLUE}[2/3] Creating PVC...${NC}"
        TMP_PVC="/tmp/k8s-policy-pvc-$$.yaml"
        sed -e "s|NAMESPACE_PLACEHOLDER|${NAMESPACE}|g" \
            "$(dirname "$0")/k8s-policy-training-pvc.yaml" > "${TMP_PVC}"
        kubectl apply -f "${TMP_PVC}"
        rm "${TMP_PVC}"
        
        # Step 3: Job
        echo -e "\n${BLUE}[3/3] Submitting job...${NC}"
        TMP_JOB="/tmp/k8s-policy-job-$$.yaml"
        sed -e "s|NAMESPACE_PLACEHOLDER|${NAMESPACE}|g" \
            -e "s|IMAGE_PLACEHOLDER|${FULL_IMAGE}|g" \
            "$(dirname "$0")/k8s-policy-training-job.yaml" > "${TMP_JOB}"
        kubectl apply -f "${TMP_JOB}"
        rm "${TMP_JOB}"
        
        echo -e "\n${GREEN}✓ Full deployment complete!${NC}"
        echo -e "${YELLOW}Monitor with: kubectl logs -f job/${JOB_NAME} -n ${NAMESPACE}${NC}"
        ;;
        
    9)
        echo -e "\n${GREEN}Exiting...${NC}"
        exit 0
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Done!${NC}"
