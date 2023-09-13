#! /usr/bin/bash

usage() {
    local ret="${1:-1}"
  echo "Usage: $0 [-h|d] [-b BUCKET_NAME] [up|down|both]"
  if ((ret == 0)); then
    echo ""
    echo "Sync exp/ with S3 bucket. Expects BUCKET_NAME variable to be defined"
    echo ""
    echo "  up = local -> s3"
    echo "  down = s3 -> local"
    echo "  both = down then up"
    echo "(deft: $direction)"
    echo ""
    echo "Options"
    echo " -h                Display this help"
    echo " -d                Dry run only"
    echo " -b BUCKET_NAME    Manually specify bucket"
  fi
  exit $ret
}

[ -f "aws_private/aws_vars.sh" ] && source "aws_private/aws_vars.sh"
source scripts/utils.sh


OPTS=( '--exclude' '**/lbi-*' '--exclude' 'slurm_logs/**' )
direction="both"

while getopts ":hdb:" opt; do
  case $opt in
    h)
      usage 0
      ;;
    d)
      OPTS+=( --dryrun )
      ;;
    b)
      BUCKET_NAME="$opt"
      ;;
    \?)
      echo "Invalid option -$OPTARG"
      usage;;
  esac
done

shift $((OPTIND - 1))

if [ ! -z "$1" ]; then
  argcheck_is_a_choice "direction" up down both "$1"
  direction="$1"
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