#!/bin/bash
set -e
date
echo
git --no-pager log -1 -b
echo
python generate_synthetic_raw_data.py -f synth_config
sh ./prepare_train_map.sh -f synth_config

