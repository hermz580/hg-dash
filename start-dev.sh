#!/bin/bash

# Exit on error
set -e

# Function to kill child processes on exit
cleanup() {
    echo "Shutting down servers..."
    if ps -p $BACKEND_PID > /dev/null
    then
       kill $BACKEND_PID
    fi
    exit
}

# Trap script exit
trap cleanup SIGINT SIGTERM

# Start the backend server in the background
echo "--- Starting backend server ---"
cd backend
# Check if requirements are installed, if not, install them
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi
uvicorn server:app --host 0.0.0.0 --port 8005 &
BACKEND_PID=$!
cd ..

# Wait a bit for the backend to start
sleep 3

# Start the frontend server
echo "--- Starting frontend server ---"
cd frontend
# Check if node_modules exist, if not, install them
if [ ! -d "node_modules" ]; then
    yarn install
fi
yarn start

# The script will hang here on `yarn start`.
# When the user presses Ctrl+C, the `cleanup` function will be called.
