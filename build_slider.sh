#!/bin/bash

# Build frontend
cd custom_components/depth_slider/frontend
npm install
npm run build

# Install Python package
cd ..
pip install -e . 