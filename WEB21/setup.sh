#!/bin/bash

# Create necessary directories
mkdir -p templates
mkdir -p uploads
mkdir -p results/images
mkdir -p static/css
mkdir -p static/js

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Activating virtual environment on Windows..."
    source venv/Scripts/activate
else
    echo "Activating virtual environment on Linux/Mac..."
    source venv/bin/activate
fi

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

echo "Setup complete! Run 'python app.py' to start the application."
echo "Access the web interface at http://localhost:5000"