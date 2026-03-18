#!/usr/bin/env python3
"""Connect to SageMaker space via SSM Session Manager."""
import subprocess
import json
import sys

ARN = "arn:aws:sagemaker:eu-central-1:698516610829:space/d-dbpzn6wx78xs/quickstart-gpu-0az0xd"
REGION = "eu-central-1"
PROFILE = "sagemaker"

# Get session from SageMaker
result = subprocess.run(
    ["aws", "sagemaker", "start-session",
     "--resource-identifier", ARN,
     "--region", REGION,
     "--profile", PROFILE],
    capture_output=True, text=True
)

if result.returncode != 0:
    print(f"Error getting session: {result.stderr}")
    sys.exit(1)

session = json.loads(result.stdout)
print(f"Got session: {session['SessionId']}")

# Build JSON for plugin (lowercase keys required)
plugin_json = json.dumps({
    "sessionId": session["SessionId"],
    "streamUrl": session["StreamUrl"],
    "tokenValue": session["TokenValue"]
})

# Call session-manager-plugin
print("Starting session...")
subprocess.run(["session-manager-plugin", plugin_json, REGION, "StartSession"])
