#!/bin/bash

# Exit on error
set -e

echo "Building depth slider component..."

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is required but not installed."
    exit 1
fi

# Navigate to frontend directory
cd custom_components/depth_slider/frontend

# Create build directory if it doesn't exist
mkdir -p build

# Install dependencies
echo "Installing frontend dependencies..."
npm install

# Build frontend
echo "Building frontend..."
npm run build

# Navigate back to component root
cd ..

# Install Python package
echo "Installing Python package..."
pip install -e .

echo "Build complete! The depth slider component is ready to use." 