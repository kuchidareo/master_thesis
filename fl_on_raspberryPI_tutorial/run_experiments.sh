#!/bin/bash

# Define the values for ATTRIBUTE1 and ATTRIBUTE2
NUM_SPLIT_VALUES=("100" "50" "20" "10")
SAMPLE_FRACTION_VALUES=("1.0" "0.5" "0.2" "0.1")

# Iterate over the arrays using index
for (( i=0; i<${#NUM_SPLIT_VALUES[@]}; i++ )); do
    for (( j=0; j<$(#SAMPLE_FRACTION_VALUES[@]); j++)) do
        NUM_SPLIT_VALUE="${NUM_SPLIT_VALUES[i]}"
        SAMPLE_FRACTION_VALUE="${SAMPLE_FRACTION_VALUES[j]}"
        # Switch to the TAB_NUMBER tab and run the server command
        xdotool key alt+1
        xdotool type --delay 1 --clearmodifiers "python3 server.py --rounds=100 --sample_fraction=${SAMPLE_FRACTION_VALUE} --min_num_clients=10"
        xdotool key Return

        xdotool key alt+2
        xdotool type --delay 1 --clearmodifiers "python3 client_pytorch.py --cid=0 --num_clients=${NUM_SPLIT_VALUE} --mnist --noniid"
        xdotool key Return

        xdotool key alt+3
        xdotool type --delay 1 --clearmodifiers "python3 client_pytorch.py --cid=1 --num_clients=${NUM_SPLIT_VALUE} --mnist --noniid"
        xdotool key Return

        xdotool key alt+4
        xdotool type --delay 1 --clearmodifiers "python3 client_pytorch.py --cid=2 --num_clients=${NUM_SPLIT_VALUE} --mnist --noniid"
        xdotool key Return

        xdotool key alt+5
        xdotool type --delay 1 --clearmodifiers "python3 client_pytorch.py --cid=3 --num_clients=${NUM_SPLIT_VALUE} --mnist --noniid"
        xdotool key Return

        xdotool key alt+6
        xdotool type --delay 1 --clearmodifiers "python3 client_pytorch.py --cid=4 --num_clients=${NUM_SPLIT_VALUE} --mnist --noniid"
        xdotool key Return

        xdotool key alt+7
        xdotool type --delay 1 --clearmodifiers "python3 client_pytorch.py --cid=5 --num_clients=${NUM_SPLIT_VALUE} --mnist --noniid"
        xdotool key Return

        xdotool key alt+8
        xdotool type --delay 1 --clearmodifiers "python3 client_pytorch.py --cid=6 --num_clients=${NUM_SPLIT_VALUE} --mnist --noniid"
        xdotool key Return

        xdotool key alt+9
        xdotool type --delay 1 --clearmodifiers "python3 client_pytorch.py --cid=7 --num_clients=${NUM_SPLIT_VALUE} --mnist --noniid"
        xdotool key Return

        xdotool key alt+0
        xdotool type --delay 1 --clearmodifiers "python3 client_pytorch.py --cid=8 --num_clients=${NUM_SPLIT_VALUE} --mnist --noniid"
        xdotool key Return

        xdotool key alt+-
        xdotool type --delay 1 --clearmodifiers "python3 client_pytorch.py --cid=9 --num_clients=${NUM_SPLIT_VALUE} --mnist --noniid"
        xdotool key Return

        if [[ "${NUM_SPLIT_VALUE}" == "10" ]]; then
            # Sleep for 1200 seconds (20 minutes)
            sleep 1200
        else
            # Sleep for 600 seconds (10 minutes)
            sleep 600
        fi
done