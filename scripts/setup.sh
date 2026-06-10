#!/bin/bash
# Setup script for FIREFLY FPV Interceptor System

echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "Setting up Node.js Ground Control Station..."
cd src/gcs
npm install

echo "Setup complete."
