#!/bin/bash

# load variables from other file
DIR=$(dirname $(realpath $0))
source "$DIR/.secret"

# The computer of cgpool
POOL=3

CGPOOL="$LOGIN@cgpool120$POOL.informatik.uni-tuebingen.de"
CGCONTACT="$LOGIN@cgcontact.informatik.uni-tuebingen.de"

# Connect and enter password
$DIR/pw.sh $PASSWORD ssh -tt $CGCONTACT "ssh -tt $CGPOOL $@"
