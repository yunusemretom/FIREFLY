#!/bin/bash
# Launch script for simulation environment

source venv/bin/activate
export SIMULATION_MODE=true
python -m src.python.simulation.scenario_runner
