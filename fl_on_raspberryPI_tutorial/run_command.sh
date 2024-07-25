#!/bin/bash

# Check if commands are provided as an argument
if [ -z "$1" ]; then
  # Prompt the user for the commands if not provided as an argument
  read -p "Enter the commands to run in each tab (separate with semicolons): " commands
else
  # Capture the commands from the argument
  commands="$1"
fi

# Open 10 new GNOME terminal tabs and run the commands in each
for i in {1..10}; do
  gnome-terminal --tab -- bash -c "$commands; exec bash"
done
