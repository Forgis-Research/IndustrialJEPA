#!/bin/bash
# Start autoresearch with environment loaded
# Usage: ./start.sh [claude args...]

cd "$(dirname "$0")/.."

# Load environment variables
set -a
source .env
set +a

echo "Environment loaded."
echo "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:0:20}..."

# Run claude with any passed arguments
cd autoresearch
exec claude "$@"
