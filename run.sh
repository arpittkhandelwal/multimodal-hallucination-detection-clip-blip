#!/bin/bash

echo "Starting MLLM Hallucination Detector..."

# Check if venv exists, if not create it and install requirements
if [ ! -d "venv" ]; then
    echo "Creating virtual environment and installing dependencies..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "Launching application..."
python3 app.py
