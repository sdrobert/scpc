#!/usr/bin/env bash

# Copyright 2023 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e

if [ ! -d "prep" ]; then
    echo "'prep' is not a directory. Did you call

    git submodule update --init"

    exit 1
fi

if [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo '$CONDA_DEFAULT_ENV set to base. This is probably undesired. If you'
    echo "really want this, modify $0 to exclude the check"
    exit 1
fi

# We likely don't need everything, but just in case
if ! pip freeze | grep 'scpc' --quiet; then
    pip install -e '.[all]'
fi

# command-line option parsing
source scripts/preamble.sh

dl="$data/librispeech"
em="$exp/$model/version_$ver"
# XXX(sdrobert): we save to experiment dir because reps are specific to the
# experiment
dlf="$em/reps"


ckpt="$em/best.ckpt"
if [ ! -f "$ckpt" ]; then
    echo "'$ckpt' is not a file (did you finish ./run.sh?)"
    exit 1
fi

# why the 960hr set if we're only training on clean 100? To ensure we have
# access to the transcripts of all partitions for LM training
if [ -z "$libri" ] && [ ! -f "$dl/.960_complete2" ]; then
    libri="$dl/local/data"
    if [ ! -f "$libri/.960_complete" ]; then
        echo "Downloading librispeech"
        $cmd_p python prep/librispeech.py "$dl" download ${TR2DL_ARGS[960]}
        touch "$libri/.960_complete"
        ((only)) && exit 0
    fi
fi

# in ./run.sh we don't generate an LM. Using a new file flag ensures we do.
# Adding the LM shouldn't mess up anything upsteam.
if [ ! -f "$dl/.960_complete2" ]; then
    echo "Performing common prep of librispeech"
    $cmd_p python prep/librispeech.py "$dl" preamble \
        --speakers-are-readers --exclude-subsets "$libri"
    $cmd_p python prep/librispeech.py "$dl" init_char "$libri" \
        --custom-lm-max-order 15 --custom-lm-prune-count 1
    touch "$dl/.960_complete2"
    ((only)) && exit 0
fi

for x in train-clean-100 dev-clean dev-other test-other; do
  y="${x//-/_}"
  pdir="$dlf/$y"
  if [ ! -f "$pdir/.a2r_complete" ]; then
    echo "Computing representations in $pdir"
    tmp="$em/tmp"
    mkdir -p "$tmp" "$pdir/feat"
    head -n 5 "$dl/local/char/$y.wav.scp" | xargs -I {} bash -c 'ln -sf "${1#* }" "${1%% *}.flac" ' -- "$tmp/"{}
    exit 1
    $cmd_p scpc-a2r $expert_args --audio-suffix .flac \
        "$libri/$x" "$ckpt" "$pdir/feat"
    touch "$pdir/.a2r_complete"
    ((only)) && exit 0
  fi
done

if [ ! -f "$dlf/.100_complete" ]; then
    echo "Converting into SpectDataSet"
    mkdir -p "$em/local"
    cp -r "$dl/local/char/" "$em/local/char" 
    $cmd_p python prep/librispeech.py \
        "$em" torch_dir char reps --feats-from "$dlf" ${TR2TD_ARGS[100]}
    touch "$dlf/.100_complete"
    ((only)) && exit 0
fi

