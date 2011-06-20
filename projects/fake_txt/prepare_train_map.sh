#!/bin/bash
set -e
ARCH=""
ARCH="arch -i386"  # for 32/64 bit issues with some MacOSX pythons
$ARCH python prepare.py "$@"
$ARCH python train.py "$@"
$ARCH python get_codes.py "$@"
$ARCH python get_coords.py "$@"
$ARCH python -i get_map.py "$@"
