#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Starting Building Simulation Demo..."
echo "=================================="

# Kill any existing processes on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null

# Start the simulation in the background
echo "Starting simulation..."
cd "$PROJECT_ROOT"
python main.py test_configs/llm_agent_demo_config.json &
SIM_PID=$!
echo "Simulation PID: $SIM_PID"

# Wait a moment for simulation to initialize
sleep 2

# Start the HTTP server from project root (to access UI_demo_outputs)
# Redirect stderr to /dev/null to suppress request logging spam
echo "Starting Demo server on http://localhost:8080..."
cd "$PROJECT_ROOT"
python -m http.server 8080 2>/dev/null &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo ""
echo "=================================="
echo "Demo is running!"
echo "Open http://localhost:8080/Demo/index.html in your browser"
echo ""
echo "Press Ctrl+C to stop both simulation and server"
echo "=================================="

# Handle Ctrl+C to clean up both processes
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $SIM_PID 2>/dev/null
    kill $SERVER_PID 2>/dev/null
    echo "Done."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for simulation to finish (not the server)
wait $SIM_PID
SIM_EXIT_CODE=$?

echo ""
echo "=================================="
echo "Simulation finished (exit code: $SIM_EXIT_CODE)"
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null
echo "Done."
