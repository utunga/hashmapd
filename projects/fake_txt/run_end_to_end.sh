#!/bin/bash
python recreate_couch_and_data.py -f orig_config
python get_raw_data_from_couch.py -f orig_config
/bin/bash ./prepare_train_map.sh -f orig_config

