#!/bin/bash
set -e
echo "Output of" $0 $@
echo
date
echo
git --no-pager log -1 | head -1
echo
python generate_synthetic_raw_data.py -f synth_config
sh ./prepare_train_map.sh -f synth_config

