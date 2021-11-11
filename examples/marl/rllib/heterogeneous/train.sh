#!/bin/bash
#
# Note:  Change the arguments based on your available resources.
#
# If running on VPN on MacOS, add the additional argument for train.py:
#    --node-ip-address $(ipconfig getifaddr en0)
clear
python -u train.py \
    --num-cpus 2 \
    --num-gpus 0 \
    --local-dir ./ray_results \
    --max-episode-steps 250

