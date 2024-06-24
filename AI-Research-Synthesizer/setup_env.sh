#!/bin/bash

# Navigate to the project directory
cd /Users/student/Desktop/SUMMER24/synth_proj

# Activate the virtual environment
source fresh_project_env/bin/activate

# Ensure all dependencies are installed
pip install -r requirements.txt

# Print a success message
echo "Environment set up successfully. You can now run 'python main.py'"
