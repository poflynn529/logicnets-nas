#!/bin/bash
# Script to start Docker container and sign in as root

CONTAINER_NAME="9cf07ecb21b9d1403ae1a2ee34664b2b6955478f11ed70c211ec96731b4fc6ef" # Replace with your container name or ID

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
