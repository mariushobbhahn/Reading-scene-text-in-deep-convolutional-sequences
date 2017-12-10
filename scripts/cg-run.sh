#!/usr/bin/env bash
DIR=$(dirname $(realpath $0))

SCRIPT=$(cat $1)

shift

$DIR/cg-login.sh "\"echo \\\"$SCRIPT\\\" | bash\""