#!/bin/bash
set -e
echo "Output of" $0 $@
echo
date
echo
git --no-pager log -1 | head -1
echo
rm -f trace/*png
ARCH=""
#ARCH="arch -i386"  # arch override for 32/64 bit issues with some MacOSX pythons (bit removed for now)
$ARCH python prepare.py
$ARCH python train.py 
$ARCH python get_codes.py
$ARCH python get_coords.py
$ARCH python get_map.py 
