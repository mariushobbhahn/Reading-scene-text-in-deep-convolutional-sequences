#!/bin/bash

# load variables from other file
DIR=$(dirname $(realpath $0))
source "$DIR/.secret"

# The computer of cgpool
POOL=1

CGPOOL="$LOGIN@cgpool120$POOL.informatik.uni-tuebingen.de"
CGCONTACT="$LOGIN@cgcontact.informatik.uni-tuebingen.de"

# Connect and enter password
$DIR/pw.sh $PASSWORD ssh -C -L 6006:127.0.0.1:6006 -tt $CGCONTACT "ssh -C -L 6006:127.0.0.1:6006 -tt $CGPOOL $@"
