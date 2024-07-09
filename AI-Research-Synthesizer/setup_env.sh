#!/bin/bash

# Navigate to the project directory
cd /path

# Activate the virtual environment
source pathenv/bin/activate

# Ensure all dependencies are installed
pip install -r requirements.txt

# Print a success message
echo "Environment set up successfully. You can now run 'python main.py'"
