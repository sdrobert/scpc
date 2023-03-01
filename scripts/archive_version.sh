#! /usr/bin/env bash

# looks in the exp and exp/tb_logs for version_0 folders and moves them to
# version_<NV>, where <NV> is the number of versions of that model

exp="${1:-"exp"}"

sources=( $(find "$exp/" -type d -name 'version_0') )
for x in "${sources[@]}"; do
  root="$(dirname "$x")"
  tb_root="$exp/tb_logs/$(basename "$root")"
  NV="$(ls -1 "$root" | wc -l)"
  if [ -d "$tb_root/version_0" ]; then
    NV_="$(ls -1 "$tb_root" | wc -l)"
    if [ "$NV" -ne "$NV_" ]; then
      echo -e "Number of versions in $root differ from those in $tb_root;" \
        "taking the greater"
      [ "$NV_" -gt "$NV" ] && NV="$NV_"
    fi
  else
    echo -e "No corresponding tensorboard logs for $x"
  fi
  echo "Moving '$x' to '$root/version_$NV'"
  mv "$x" "$root/version_$NV"
  if [ -d "$tb_root/version_0" ]; then
    echo "Moving '$tb_root/version_0' to '$tb_root/version_$NV'"
    mv "$tb_root/version_0" "$tb_root/version_$NV"
  fi
done

for x in $(find "$exp/tb_logs" -type d -name 'version_0'); do
  echo -e "Unmatched tensorboard log dir '$x'"
done
