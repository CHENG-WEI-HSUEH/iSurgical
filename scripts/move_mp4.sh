#!/bin/bash

# ==========================================
# Remote Server and Path Configuration
# ==========================================
REMOTE_USER="pandora"
REMOTE_IP="127.0.0.1"
REMOTE_PATH="/mnt/win_share"

# Relies on SSH Key passwordless login (no longer using plaintext passwords / sshpass).
# Please ensure the environment running this script has added the corresponding private key 
# to the SSH agent or ~/.ssh, and the public key has been added to 
# ${REMOTE_USER}@${REMOTE_IP}:~/.ssh/authorized_keys.

# ==========================================
# Dynamically Get Script Argument
# ==========================================
# $1 represents the first argument entered when running the script
TARGET_FILE="$1"

# 1. Check if the user provided a filename argument
if [ -z "$TARGET_FILE" ]; then
    echo -e "\033[31m[Error] Please provide the filename to transfer!\033[0m"
    echo -e "Usage example: \033[36m$0 filename.mp4\033[0m"
    exit 1
fi

# 2. Check if the local target video file exists
if [ ! -f "$TARGET_FILE" ]; then
    echo -e "\033[31m[Error] File not found in the current directory: $TARGET_FILE\033[0m"
    exit 1
fi

echo "------------------------------------------------"
echo "File to transfer: $TARGET_FILE"
echo "Target Server:    $REMOTE_IP"
echo "Target Path:      $REMOTE_PATH"
echo "------------------------------------------------"

# 3. Execute transfer (relies on SSH Key passwordless login)
rsync -avP -e "ssh -o StrictHostKeyChecking=no" --rsync-path="mkdir -p $REMOTE_PATH && rsync" "$TARGET_FILE" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}"

# 4. Check transfer exit code
if [ $? -eq 0 ]; then
    echo -e "\033[32m\n[Success] File $TARGET_FILE has been successfully transferred!\033[0m"
else
    echo -e "\033[31m\n[Failure] Transfer incomplete. Please check your network or permissions.\033[0m"
    exit 2
fi