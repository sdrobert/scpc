#! /usr/bin/env bash

# Remove any other .ckpt files in exp/ subdirectories which already have a
# best.ckpt

find exp/ -mindepth 3 -maxdepth 3 -name 'best.ckpt' -exec dirname {} \; |
  xargs -I % find % -maxdepth 1 -name '*.ckpt' -not -name 'best.ckpt' -delete
