#!/bin/bash

# ========================================
# A2A-ACP Push Notifications Deployment Script
# ========================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."

    if ! command_exists curl; then
        print_error "curl is required but not installed."
        exit 1
    fi

    if ! command_exists docker; then
        print_warning "Docker not found. Installing Docker..."

        # Install Docker (basic installation for Linux)
        if command_exists apt-get; then
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
            print_success "Docker installed successfully"
        else
            print_error "Please install Docker manually: https://docs.docker.com/get-docker/"
            exit 1
        fi
    else
        print_success "Dependencies check passed"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."

    mkdir -p data
    mkdir -p logs

    print_success "Directories created"
}

# Generate .env file if it doesn't exist
setup_environment() {
    if [ ! -f .env ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env

        # Generate a secure random token
        if command_exists openssl; then
            A2A_TOKEN=$(openssl rand -hex 32)
            sed -i "s/A2A_AUTH_TOKEN=.*/A2A_AUTH_TOKEN=\"$A2A_TOKEN\"/" .env
        fi

        print_warning ".env file created. Please review and customize the settings."
        print_warning "Your A2A_AUTH_TOKEN has been generated automatically."
    else
        print_success ".env file already exists"
    fi
}

# Build and deploy with Docker Compose
deploy_with_docker() {
    print_status "Building and deploying with Docker Compose..."

    # Stop existing containers
    docker-compose down || true

    # Build and start services
    docker-compose up --build -d

    print_success "Application deployed successfully!"
    print_status "Application is running at: http://localhost:8000"
    print_status "Health check available at: http://localhost:8000/health"
    print_status "Push notification metrics at: http://localhost:8000/metrics/push-notifications"
    print_status "System metrics at: http://localhost:8000/metrics/system"
}

# Check if application is healthy
check_health() {
    print_status "Checking application health..."

    # Wait for application to start
    print_status "Waiting for application to be ready..."
    sleep 10

    # Check health endpoint
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_success "Application is healthy!"

        # Display service information
        echo
        echo "=========================================="
        echo "A2A-ACP Push Notifications is running!"
        echo "=========================================="
        echo "API Endpoints:"
        echo "  - Main API:        http://localhost:8000"
        echo "  - Health Check:    http://localhost:8000/health"
        echo "  - Metrics (Push):  http://localhost:8000/metrics/push-notifications"
        echo "  - Metrics (System): http://localhost:8000/metrics/system"
        echo "  - WebSocket:       ws://localhost:8000/streaming/websocket"
        echo "  - Server-Sent Events: http://localhost:8000/streaming/sse"
        echo
        echo "A2A Protocol Endpoints:"
        echo "  - JSON-RPC:        http://localhost:8000/a2a/rpc"
        echo "  - Message Send:    http://localhost:8000/a2a/message/send"
        echo
        echo "Logs are available in: ./logs/"
        echo "Data is stored in: ./data/"
        echo
        echo "To stop the application: docker-compose down"
        echo "To view logs: docker-compose logs -f"
        echo "=========================================="
    else
        print_error "Application health check failed!"
        print_error "Check the logs with: docker-compose logs"
        exit 1
    fi
}

# Main deployment flow
main() {
    echo "=========================================="
    echo "A2A-ACP Push Notifications Deployment"
    echo "=========================================="
    echo

    check_dependencies
    create_directories
    setup_environment
    deploy_with_docker
    check_health

    print_success "Deployment completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "stop")
        print_status "Stopping application..."
        docker-compose down
        print_success "Application stopped"
        ;;
    "logs")
        print_status "Showing application logs..."
        docker-compose logs -f
        ;;
    "restart")
        print_status "Restarting application..."
        docker-compose restart
        print_success "Application restarted"
        ;;
    "status")
        print_status "Checking application status..."
        docker-compose ps
        ;;
    *)
        main
        ;;
esac