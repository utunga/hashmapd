#!/bin/bash
python generate_synthetic_raw_data.py -f synth_config
arch sh ./prepare_train_map.sh -f synth_config

