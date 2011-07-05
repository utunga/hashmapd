#!/bin/bash -e

GIT_ROOT=$(git rev-parse --show-toplevel)
CANVAS_DIR=$GIT_ROOT/website/www/canvas

TOKEN_DATA=$CANVAS_DIR/tokens/all-tokens-7.json
USER_DATA=$CANVAS_DIR/locations.json
#TOKEN_DATA=$CANVAS_DIR/tokens/all-tokens-10.json.gz
#USER_DATA=$CANVAS_DIR/tokens/users-9.json


rm -f labels.pickle

BASE=$(basename ${TOKEN_DATA%*.json*})

if [[ "$1" ]]; then
    TARGETS=$1
else
    TARGETS='magic token_total token_relative total adjusted relative'
fi

for x in $TARGETS; do
    echo $BASE-$x.html
    time ./find_labels.py $TOKEN_DATA $USER_DATA $CANVAS_DIR/$BASE-$x.html $x
done
