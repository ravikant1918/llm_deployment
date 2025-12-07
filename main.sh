#!/bin/bash
# Interactive LLM Deployment Script
# One-command setup and deployment of TensorRT-LLM or vLLM

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}🚀 INTERACTIVE LLM DEPLOYMENT${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_step() {
    echo -e "${CYAN}[STEP $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${PURPLE}ℹ${NC} $1"
}

# Check if running on compute node
check_environment() {
    print_step "1" "Checking Environment"

    if [[ $(hostname) != exp-blr-dgxb200-01 ]]; then
        print_warning "Not running on compute node exp-blr-dgxb200-01"
        echo "Current host: $(hostname)"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Running on compute node exp-blr-dgxb200-01"
    fi
}

# Setup .env configuration
setup_config() {
    print_step "2" "Configuration Setup"

    if [ ! -f ".env" ]; then
        print_info "Creating .env file from template..."
        cp .env.example .env
        print_success ".env file created"
    else
        print_info ".env file already exists"
    fi

    echo
    echo -e "${WHITE}Current configuration:${NC}"
    echo "----------------------------------------"
    if [ -f ".env" ]; then
        grep -E "^(NGC_API_KEY|MODEL_NAME|FRAMEWORK|PORT)=" .env | head -10
    fi
    echo "----------------------------------------"

    echo
    read -p "Do you want to edit the .env file? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ${EDITOR:-nano} .env
        print_success "Configuration updated"
    fi

    # Validate critical settings
    source scripts/setup/load_config.sh

    if [ "$NGC_API_KEY" == "your-ngc-api-key-here" ] || [ -z "$NGC_API_KEY" ]; then
        print_error "NGC_API_KEY not configured!"
        echo "Please edit .env and set your NGC API key from: https://ngc.nvidia.com/setup/api-key"
        exit 1
    fi

    print_success "Configuration validated"
}

# Choose deployment framework
choose_framework() {
    print_step "3" "Framework Selection"

    echo
    echo -e "${WHITE}Available Frameworks:${NC}"
    echo "1) TensorRT-LLM (Recommended) - 2-10x faster, production-ready"
    echo "2) vLLM - Easier setup, good performance, more flexible"
    echo

    while true; do
        read -p "Choose framework (1-2): " choice
        case $choice in
            1)
                FRAMEWORK="tensorrt-llm"
                break
                ;;
            2)
                FRAMEWORK="vllm"
                break
                ;;
            *)
                print_warning "Please choose 1 or 2"
                ;;
        esac
    done

    print_success "Selected: $FRAMEWORK"

    # Update .env with chosen framework
    sed -i.bak "s/^FRAMEWORK=.*/FRAMEWORK=\"$FRAMEWORK\"/" .env
}

# Setup directories
setup_directories() {
    print_step "4" "Directory Setup"

    print_info "Creating standard directory structure..."
    bash scripts/setup/setup_directories.sh

    print_success "Directories created"
}

# Download model
download_model() {
    print_step "5" "Model Download"

    source scripts/setup/load_config.sh

    if [ "$AUTO_DOWNLOAD_MODEL" != "true" ]; then
        print_info "Auto-download disabled in .env"
        read -p "Download model now? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Skipping model download"
            return
        fi
    fi

    print_info "Downloading model: $MODEL_NAME"
    bash scripts/download/download_model_auto.sh

    print_success "Model ready"
}

# Setup container
setup_container() {
    print_step "6" "Container Setup"

    source scripts/setup/load_config.sh

    print_info "Setting up $FRAMEWORK container..."
    bash deploy/02_setup_container.sh

    print_success "Container ready"
}

# Start deployment
start_deployment() {
    print_step "7" "Starting Deployment"

    source scripts/setup/load_config.sh

    print_info "Starting $FRAMEWORK container..."
    bash deploy/03_start_container.sh

    print_success "Container started"
    echo
    echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE!${NC}"
    echo
    echo -e "${WHITE}Your LLM is now running!${NC}"
    echo
    echo -e "${CYAN}API Endpoint:${NC} http://localhost:$PORT"
    echo -e "${CYAN}Model:${NC} $MODEL_NAME"
    echo -e "${CYAN}Framework:${NC} $FRAMEWORK"
    echo
    echo -e "${YELLOW}Test the API:${NC}"
    echo "curl -X POST http://localhost:$PORT/v1/completions \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"model\": \"$MODEL_NAME\", \"prompt\": \"Hello, how are you?\", \"max_tokens\": 50}'"
    echo
    echo -e "${YELLOW}Run benchmarks:${NC}"
    echo "python3 scripts/benchmark/benchmark_tensorrt_llm.py --server-url http://localhost:$PORT"
}

# Test deployment
test_deployment() {
    print_step "8" "Testing Deployment"

    source scripts/setup/load_config.sh

    print_info "Running quick server test..."
    bash scripts/test/quick_test_server.sh

    print_success "Server test completed"
}

# Main menu
show_menu() {
    echo
    echo -e "${WHITE}What would you like to do?${NC}"
    echo "1) 🚀 Full deployment (setup everything and run)"
    echo "2) 🔧 Setup only (configure environment)"
    echo "3) ▶️  Deploy only (run existing setup)"
    echo "4) 🧪 Test only (test running server)"
    echo "5) 📊 Benchmark only (performance testing)"
    echo "6) ❌ Exit"
    echo
}

# Full deployment
full_deployment() {
    check_environment
    setup_config
    choose_framework
    setup_directories
    download_model
    setup_container
    start_deployment
    test_deployment
}

# Setup only
setup_only() {
    check_environment
    setup_config
    choose_framework
    setup_directories
    download_model
    setup_container
    print_success "Setup complete! Run 'bash main.sh' again to deploy."
}

# Deploy only
deploy_only() {
    check_environment
    start_deployment
    test_deployment
}

# Test only
test_only() {
    check_environment
    test_deployment
}

# Benchmark only
benchmark_only() {
    source scripts/setup/load_config.sh

    if [ ! -d "scripts/benchmark" ]; then
        print_error "Benchmark scripts not found"
        exit 1
    fi

    print_info "Installing benchmark dependencies..."
    bash scripts/benchmark/install_benchmark_deps.sh

    print_info "Running comprehensive benchmark..."
    python3 scripts/benchmark/benchmark_tensorrt_llm.py --server-url "http://localhost:$PORT"

    print_success "Benchmarking complete"
}

# Main function
main() {
    print_header

    while true; do
        show_menu
        read -p "Choose option (1-6): " choice

        case $choice in
            1)
                echo
                print_info "Starting FULL DEPLOYMENT..."
                full_deployment
                break
                ;;
            2)
                echo
                print_info "Starting SETUP ONLY..."
                setup_only
                break
                ;;
            3)
                echo
                print_info "Starting DEPLOY ONLY..."
                deploy_only
                break
                ;;
            4)
                echo
                print_info "Starting TEST ONLY..."
                test_only
                break
                ;;
            5)
                echo
                print_info "Starting BENCHMARK ONLY..."
                benchmark_only
                break
                ;;
            6)
                echo
                print_info "Goodbye!"
                exit 0
                ;;
            *)
                print_warning "Invalid option. Please choose 1-6."
                ;;
        esac
    done
}

# Run main function
main "$@"