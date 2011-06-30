#!/bin/bash -e

TOKEN_DATA=website/www/canvas/tokens/all-tokens-7.json
USER_DATA=website/www/canvas/locations.json
#TOKEN_DATA=website/www/canvas/tokens/all-tokens-10.json.gz
#USER_DATA=website/www/canvas/tokens/users-9.json


rm -f labels.pickle

BASE=$(basename ${TOKEN_DATA%*.json*})

if [[ "$1" ]]; then
    TARGETS=$1
else
    TARGETS='magic token_total token_relative total adjusted relative'
fi

for x in $TARGETS; do
    echo $BASE-$x.html
    time ./find_labels.py $TOKEN_DATA $USER_DATA website/www/canvas/$BASE-$x.html $x
done
