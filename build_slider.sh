#!/bin/bash

# Exit on error
set -e

echo "Building depth slider component..."

# Set Node.js environment for compatibility
export NODE_OPTIONS=--openssl-legacy-provider

cd custom_components/depth_slider/frontend

# Install dependencies
echo "Installing frontend dependencies..."
npm install

# Build frontend
echo "Building frontend..."
npm run build

# Go back to component root
cd ..

# Install Python package
echo "Installing Python package..."
pip install -e .

echo "Build complete! The depth slider component is ready to use." 