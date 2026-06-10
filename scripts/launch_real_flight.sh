#!/bin/bash
# Launch script for real autonomous flight

source venv/bin/activate
export SIMULATION_MODE=false
python -m src.python.guidance.mode_manager
