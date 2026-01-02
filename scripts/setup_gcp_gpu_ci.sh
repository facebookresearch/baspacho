#!/bin/bash
# BaSpaCho GCP GPU CI Infrastructure Setup
# This script sets up the GCP resources needed for GPU testing via Cloud Run
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GitHub CLI (gh) installed and authenticated (optional, for auto-setting secrets)
#   - A billing account linked to your GCP organization
#
# Usage:
#   ./setup_gcp_gpu_ci.sh [OPTIONS]
#
# Options:
#   --project PROJECT_ID    GCP project ID (default: baspacho-gpu-ci)
#   --region REGION         GCP region (default: us-central1)
#   --github-repo OWNER/REPO  GitHub repository (default: detected from git origin)
#   --billing-account ID    Billing account ID (required if creating new project)
#
# This script is fully idempotent - safe to run multiple times.

set -euo pipefail

# Default configuration
PROJECT_ID="${GCP_PROJECT:-baspacho-gpu-ci}"
REGION="${GCP_REGION:-us-central1}"
GITHUB_REPO="${GITHUB_REPO:-}"
BILLING_ACCOUNT="${BILLING_ACCOUNT:-}"
SERVICE_ACCOUNT_NAME="github-gpu-ci"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --github-repo)
            GITHUB_REPO="$2"
            shift 2
            ;;
        --billing-account)
            BILLING_ACCOUNT="$2"
            shift 2
            ;;
        --help|-h)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Detect GitHub repo from git origin if not specified
if [ -z "$GITHUB_REPO" ]; then
    ORIGIN_URL=$(git remote get-url origin 2>/dev/null || true)
    if [ -n "$ORIGIN_URL" ]; then
        # Handle both SSH and HTTPS URLs, with or without .git suffix
        GITHUB_REPO=$(echo "$ORIGIN_URL" | sed -E 's#^(https://github.com/|git@github.com:)##' | sed 's/\.git$//')
    fi
    if [ -z "$GITHUB_REPO" ]; then
        log_error "Could not detect GitHub repository. Please specify with --github-repo OWNER/REPO"
        exit 1
    fi
fi

echo "=== BaSpaCho GCP GPU CI Setup ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "GitHub Repo: $GITHUB_REPO"
echo ""

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI is not installed. Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -1 | grep -q .; then
    log_error "Not authenticated to GCP. Run: gcloud auth login"
    exit 1
fi

# ============================================================================
# Project Setup
# ============================================================================
echo "=== Checking GCP Project ==="

if gcloud projects describe "$PROJECT_ID" > /dev/null 2>&1; then
    log_info "Project already exists: $PROJECT_ID"
else
    log_info "Project does not exist, creating: $PROJECT_ID"

    # Check for billing account
    if [ -z "$BILLING_ACCOUNT" ]; then
        # Try to find a billing account
        BILLING_ACCOUNT=$(gcloud billing accounts list --filter="open=true" --format="value(name)" --limit=1 2>/dev/null || true)
        if [ -z "$BILLING_ACCOUNT" ]; then
            log_error "No billing account found. Please specify with --billing-account ACCOUNT_ID"
            log_info "List available billing accounts with: gcloud billing accounts list"
            exit 1
        fi
        log_info "Using billing account: $BILLING_ACCOUNT"
    fi

    # Create the project
    gcloud projects create "$PROJECT_ID" --name="BaSpaCho GPU CI" --quiet

    # Link billing account
    log_info "Linking billing account..."
    gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT"
fi

# Set the project as active and update quota project
gcloud config set project "$PROJECT_ID" --quiet
gcloud auth application-default set-quota-project "$PROJECT_ID" --quiet 2>/dev/null || true

echo ""

# ============================================================================
# Enable APIs
# ============================================================================
echo "=== Enabling GCP APIs ==="
APIS=(
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
    "secretmanager.googleapis.com"
    "iam.googleapis.com"
    "compute.googleapis.com"
    "cloudresourcemanager.googleapis.com"
)

for api in "${APIS[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" 2>/dev/null | grep -q .; then
        log_info "$api (already enabled)"
    else
        log_info "Enabling $api..."
        gcloud services enable "$api" --quiet
    fi
done
echo ""

# ============================================================================
# Service Account Setup
# ============================================================================
echo "=== Setting up Service Account ==="
SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "$SA_EMAIL" > /dev/null 2>&1; then
    log_info "Service account already exists: $SA_EMAIL"
else
    log_info "Creating service account: $SERVICE_ACCOUNT_NAME"
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --description="GitHub Actions GPU CI" \
        --display-name="GitHub GPU CI" \
        --quiet
fi

# Grant required roles (idempotent - add-iam-policy-binding is safe to repeat)
log_info "Ensuring IAM roles..."
ROLES=(
    "roles/run.admin"
    "roles/artifactregistry.admin"
    "roles/iam.serviceAccountUser"
    "roles/logging.viewer"
    "roles/storage.admin"
)

# Get current policy once to check existing bindings
CURRENT_POLICY=$(gcloud projects get-iam-policy "$PROJECT_ID" --format=json 2>/dev/null || echo "{}")

for role in "${ROLES[@]}"; do
    if echo "$CURRENT_POLICY" | grep -q "\"role\": \"$role\"" && echo "$CURRENT_POLICY" | grep -q "$SA_EMAIL"; then
        log_info "  $role (already granted)"
    else
        log_info "  Granting $role..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SA_EMAIL" \
            --role="$role" \
            --condition=None \
            --quiet > /dev/null 2>&1
    fi
done
echo ""

# ============================================================================
# Artifact Registry Setup
# ============================================================================
echo "=== Setting up Artifact Registry ==="
AR_REPO="ghcr-remote"

if gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" > /dev/null 2>&1; then
    log_info "Artifact Registry repository already exists: $AR_REPO"
else
    log_info "Creating Artifact Registry remote repository..."
    gcloud artifacts repositories create "$AR_REPO" \
        --repository-format=docker \
        --location="$REGION" \
        --mode=remote-repository \
        --remote-docker-repo=https://ghcr.io \
        --description="Remote repository for GHCR images" \
        --quiet
fi
echo ""

# ============================================================================
# sccache GCS Bucket Setup
# ============================================================================
echo "=== Setting up sccache GCS Bucket ==="
SCCACHE_BUCKET="baspacho-sccache"

if gsutil ls -b "gs://$SCCACHE_BUCKET" > /dev/null 2>&1; then
    log_info "sccache bucket already exists: $SCCACHE_BUCKET"
else
    log_info "Creating sccache GCS bucket..."
    gsutil mb -l "$REGION" "gs://$SCCACHE_BUCKET"
fi

# Grant access to the service account and default compute SA
log_info "Ensuring bucket permissions..."
gsutil iam ch "serviceAccount:$SA_EMAIL:objectAdmin" "gs://$SCCACHE_BUCKET" 2>/dev/null || true
# Also grant to default compute service account (used by Cloud Run)
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)' 2>/dev/null || echo "")
if [ -n "$PROJECT_NUMBER" ]; then
    COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    gsutil iam ch "serviceAccount:$COMPUTE_SA:objectAdmin" "gs://$SCCACHE_BUCKET" 2>/dev/null || true
fi
echo ""

# ============================================================================
# Service Account Key Management
# ============================================================================
echo "=== Managing Service Account Key ==="
SECRET_NAME="github-gpu-ci-sa-key"
SA_KEY=""

if gcloud secrets describe "$SECRET_NAME" > /dev/null 2>&1; then
    log_info "Secret already exists: $SECRET_NAME"

    # Try to get the latest version
    SA_KEY=$(gcloud secrets versions access latest --secret="$SECRET_NAME" 2>/dev/null || echo "")

    if [ -z "$SA_KEY" ] || [ "$SA_KEY" = "{}" ]; then
        log_warn "No valid secret version found, creating new key..."
        SA_KEY=$(gcloud iam service-accounts keys create /dev/stdout \
            --iam-account="$SA_EMAIL" 2>/dev/null)

        log_info "Storing new key in Secret Manager..."
        echo "$SA_KEY" | gcloud secrets versions add "$SECRET_NAME" --data-file=- --quiet
    else
        log_info "Using existing key from Secret Manager"
    fi
else
    log_info "Creating Secret Manager secret: $SECRET_NAME"
    gcloud secrets create "$SECRET_NAME" \
        --replication-policy="automatic" \
        --quiet

    log_info "Creating service account key..."
    SA_KEY=$(gcloud iam service-accounts keys create /dev/stdout \
        --iam-account="$SA_EMAIL" 2>/dev/null)

    log_info "Storing key in Secret Manager..."
    echo "$SA_KEY" | gcloud secrets versions add "$SECRET_NAME" --data-file=- --quiet
fi
echo ""

# ============================================================================
# GitHub Secret Sync
# ============================================================================
echo "=== Syncing to GitHub Secrets ==="
if command -v gh &> /dev/null; then
    # Check if we can access the repo
    if gh repo view "$GITHUB_REPO" > /dev/null 2>&1; then
        log_info "Setting GitHub secret: GCP_SERVICE_ACCOUNT_KEY"
        echo "$SA_KEY" | gh secret set GCP_SERVICE_ACCOUNT_KEY --repo="$GITHUB_REPO"
        log_info "GitHub secret set successfully!"
    else
        log_warn "Cannot access GitHub repo: $GITHUB_REPO"
        log_warn "You may need to authenticate: gh auth login"
        log_info "Manual setup required (see below)"
    fi
else
    log_warn "GitHub CLI (gh) not found"
    log_info "Install from: https://cli.github.com/"
    log_info ""
    log_info "Manual setup required:"
    log_info "  1. Go to: https://github.com/$GITHUB_REPO/settings/secrets/actions"
    log_info "  2. Create secret: GCP_SERVICE_ACCOUNT_KEY"
    log_info "  3. Get value with: gcloud secrets versions access latest --secret=$SECRET_NAME"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=== Setup Complete ==="
echo ""
echo "Resources created/verified:"
echo "  - GCP Project: $PROJECT_ID"
echo "  - Service Account: $SA_EMAIL"
echo "  - Artifact Registry: $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO"
echo "  - Secret Manager: $SECRET_NAME"
echo ""
echo "GitHub Actions Configuration:"
echo "  - Repository: $GITHUB_REPO"
echo "  - Secret: GCP_SERVICE_ACCOUNT_KEY"
echo ""
echo "Workflow environment variables (update .github/workflows/test-gpu-cloudrun.yml if needed):"
echo "  GCP_PROJECT: $PROJECT_ID"
echo "  GCP_REGION: $REGION"
echo "  AR_REMOTE_REPO: $AR_REPO"
echo ""
echo "Next steps:"
echo "  1. Commit and push the workflow files"
echo "  2. The GPU tests will run on push to main or PR"
echo ""
echo "To manually trigger GPU tests:"
echo "  gh workflow run test-gpu-cloudrun.yml --repo $GITHUB_REPO"
