#!/bin/bash

# Array of server hosts
SERVER_HOSTS=(
	"192.168.0.112"
    "192.168.0.113"
	"192.168.0.114"
	"192.168.0.115"
	"192.168.0.116"
	"192.168.0.117"
	"192.168.0.118"
	"192.168.0.119"
	"192.168.0.120"
	"192.168.0.121"	
)

# SSH details
SSH_USER="rasheed"
SSH_PASSWORD="modenaottun"

for HOST in "${SERVER_HOSTS[@]}"; do
  gnome-terminal --tab -- bash -c "
    sshpass -p '$SSH_PASSWORD' ssh -t $SSH_USER@$HOST '
      cd kuchida/master_thesis/fl_on_raspberryPI_tutorial && \
      source venv/bin/activate && \
      exec bash
    '
  "
done
