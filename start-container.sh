#!/bin/bash
# Script to start Docker container and sign in as root

CONTAINER_NAME="c9403f9c0986aaa522ec55c2401d6cb3cc0f9bd531fc54e3beea2fff125e781d" # Replace with your container name or ID

# Check if the container is running
if ! winpty docker ps | grep -q $CONTAINER_NAME; then
    echo "Starting the container: $CONTAINER_NAME"
    winpty docker start $CONTAINER_NAME
else
    echo "Container is already running."
fi

# Sign in as root
echo "Signing in as root..."
winpty docker exec -u root -it $CONTAINER_NAME //bin/bash

# Change file permissions of the workspace
# chmod -R 777 /workspace
