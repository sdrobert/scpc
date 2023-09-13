#! /usr/bin/env bash

if [ $# = 0 ]; then
  echo "Call a command repeatedly by enumerating a grid of options"
  echo "Usage: $0 <cmd> [<flag-1> [<flag-1-arg-1> [<flag-1-arg-2> ...]]] [<flag-2> ...]"
  echo "e.g."
  echo "  $0 ./run.sh -m cpc.csa{2,4,8} -v 1 2 3"
  echo "calls, in sequence:"
  echo "  ./run.sh -m cpc.csa2 -v 1"
  echo "  ./run.sh -m cpc.csa2 -v 2"
  echo "  ./run.sh -m cpc.csa2 -v 3"
  echo "  ./run.sh -m cpc.csa4 -v 1"
  echo "  ./run.sh -m cpc.csa4 -v 2"
  echo "  ./run.sh -m cpc.csa4 -v 3"
  echo "  ./run.sh -m cpc.csa8 -v 1"
  echo "  ./run.sh -m cpc.csa8 -v 2"
  echo "  ./run.sh -m cpc.csa8 -v 3"
  echo "This script will exit on the first failure"
  echo ""
  echo "WARNING: commands will be called without escaping, making this "
  echo "script vulnerable to injection attacks."
  exit 1
fi


base_cmd="$1"
shift

cur_flag=
declare -A flag2vals

for arg in "$@"; do
  if [ "${#arg}" = 2 ] && [ "${arg:0:1}" = "-" ]; then
    cur_flag="${arg:1:2}"
  else
    if [ -z "$cur_flag" ]; then
      echo "first argument must be a flag"
      exit 1
    fi
    flag2vals[$cur_flag]="${flag2vals[$cur_flag]}$arg "
  fi
done

a=( "$base_cmd" )
for flag in "${!flag2vals[@]}"; do
  b=()
  for val in ${flag2vals[${flag}]}; do
    for c in "${a[@]}"; do
      b+=( "$c -$flag $val" )
    done
  done
  a=( "${b[@]}" )
done

echo "The grid (starts in 10 secs):"
for cmd in "${a[@]}"; do
  echo "$cmd"
done
sleep 10

for cmd in "${a[@]}"; do
  $cmd || exit 1
done
