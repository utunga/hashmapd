#!/bin/bash -e


if [[ "$1" ]]; then
    TARGETS=$1
else
    TARGETS='token_total token_relative total adjusted relative'
fi

for x in $TARGETS; do
    echo $x
  time ./find_labels.py all-tokens-7.json website/www/canvas/all-tokens-$x.html $x
done
