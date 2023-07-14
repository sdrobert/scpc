#! /usr/bin/bash

usage() {
    local ret="${1:-1}"
  echo "Usage: $0 [-h] [up|down|both]"
  if ((ret == 0)); then
    echo ""
    echo "Sync exp/ with S3 bucket. Expects BUCKET_NAME variable to be defined"
    echo ""
    echo "  up = local -> s3"
    echo "  down = s3 -> local"
    echo "  both = down then up"
    echo "(deft: $direction)"
  fi
  exit $ret
}

[ -f "aws_private/aws_vars.sh" ] && source "aws_private/aws_vars.sh"
source scripts/utils.sh

OPTS=( '--exclude' 'lbi-*-.pt' '--exclude' '*.npy' )
direction="both"

if [ ! -z "$1" ]; then
  [ "$1" = "-h" ] && usage 0
  argcheck_is_a_choice "" up down both "$1"
fi

if [ -z "$BUCKET_NAME" ]; then
  echo "Error: BUCKET_NAME empty!"
  usage
fi

if [ "$direction" != "up" ]; then
  aws s3 sync "${OPTS[@]}" "s3://${BUCKET_NAME}/exp/" exp/ || \
    echo 'sync down failed'
fi
if [ "$direction" != "down" ]; then
  aws s3 sync "${OPTS[@]}" exp/ "s3://${BUCKET_NAME}/exp/" || \
    echo 'sync up failed'
fi