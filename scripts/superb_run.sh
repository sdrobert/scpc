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

# command-line option parsing
help_message="Evaluate pre-trained model on SUPERB phone rec task"
source scripts/preamble.sh

ft="${MDL2FT[$model]}"
em="$exp/$model/version_$ver"
dl="$data/librispeech"
sp="$em/superb/pr"

mdl="$em/best.ckpt"
if [ ! -f "$mdl" ]; then
    echo "Model file '$mdl' doesn't exist. Did you set -$MDL_FLG, -$VER_FLG" \
         "correctly?"
    exit 1
fi

if [ ! -d "s3prl" ]; then
    echo "'s3prl' is not a directory. Did you call

    git submodule update --init"

    exit 1
fi

if ! pip freeze | grep 's3prl' --quiet; then
    pip install -e ./s3prl
fi

if [ -z "$libri" ]; then
    libri="$dl/local/data"
    if [ ! -f "$libri/.complete" ]; then
        echo "Downloading librispeech"
        $cmd_p python prep/librispeech.py "$dl" download \
            --files \
                {test,dev}-{clean,other}.tar.gz \
                train-clean-100.tar.gz \
                librispeech-vocab.txt
        touch "$libri/.complete"
        ((only)) && exit 0
    fi
fi

if [ ! -f "$sp/config.yaml" ]; then
    mkdir -p "$sp"
    {
        export libri nwork
        cat "conf/superb-pr.yaml" | envsubst
    } > "$sp/config.yaml"
    ((only)) && exit 0
fi

$cmd python s3prl/s3prl/run_downstream.py ${FT2SUPERB_ARGS[$ft]} \
    -u scpc_local \
    -p "$sp" \
    -c "$sp/config.yaml" \
    -k "$mdl" \
    -m train \
    -d ctc \
    -a