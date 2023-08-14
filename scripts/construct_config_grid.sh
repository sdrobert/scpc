#! /usr/bin/env bash

# Useful if you've set up a number of configurations and you want to save them
# to the exp/ folder without running anything
#
# E.g.
#   $0 -m cpc.csa{2,4,8,16,32} -v 1 2 3
# generates
#   exp/cpc.csa{2,4,8,16,32}/version_{1,2,3}/model.yaml

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

a=( "${RUN_CMD:-./scripts/preamble.sh}" )
for flag in "${!flag2vals[@]}"; do
  b=()
  for val in ${flag2vals[${flag}]}; do
    for c in "${a[@]}"; do
      b+=( "$c -$flag $val" )
    done
  done
  a=( "${b[@]}" )
done

for cmd in "${a[@]}"; do
  $cmd || exit 1
done
