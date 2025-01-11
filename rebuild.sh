#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${RED}✗ $1${NC}"
        exit 1
    fi
}

# Parse command line arguments
DEBUG_MODE=0
while [ $# -gt 0 ]; do
    case $1 in
        --debug)
            DEBUG_MODE=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo -e "${YELLOW}Starting rebuild process...${NC}"

# If we're in the build directory, go up one level
if [ "$(basename $(pwd))" = "build" ]; then
    cd ..
fi

# Clean build directory
echo -e "${YELLOW}Cleaning build directory...${NC}"
if [ -d "build" ]; then
    rm -rf build
    check_status "Removed old build directory"
fi

mkdir build
check_status "Created fresh build directory"

# Navigate to build directory and run cmake
cd build
echo -e "${YELLOW}Running CMake...${NC}"

# Apply debug flags if in debug mode
if [ $DEBUG_MODE -eq 1 ]; then
    cmake -DCMAKE_CUDA_FLAGS="${CMAKE_CUDA_FLAGS} -g -G" ..
else
    cmake ..
fi
check_status "CMake configuration"

# Build the project
echo -e "${YELLOW}Building project...${NC}"
make
check_status "Project build"

# Run tests if they exist
if [ -f "kernel_tests" ]; then
    echo -e "${YELLOW}Running kernel tests...${NC}"
    ./kernel_tests
    check_status "Kernel tests"
fi

# # Run main program if it exists
# if [ -f "bozo_sort" ]; then
#     echo -e "${YELLOW}Running main program...${NC}"
#     ./bozo_sort
#     check_status "Main program execution"
# fi

echo -e "${GREEN}Rebuild complete!${NC}"
