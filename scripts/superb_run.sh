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

if [ ! -d "s3prl" ]; then
    echo "'s3prl' is not a directory. Did you call

    git submodule update --init"

    exit 1
fi

if ! pip freeze | grep 's3prl' --quiet; then
    pip install -e './s3prl'
fi

if [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo '$CONDA_DEFAULT_ENV set to base. This is probably undesired. If you'
    echo "really want this, modify $0 to exclude the check"
    exit 1
fi

# command-line option parsing
help_message="Evaluate pre-trained model on a SUPERB downstream task"
source scripts/preamble.sh

dl="$data/librispeech"
if [ -z "$pca" ]; then
  sp="$em/superb/$stask/full"
else
  sp="$em/superb/$stask/pca_$pca"
fi


if [[ "$ft" =~ ^superb. ]]; then
    superb_flags=( "-u" "${ft:7}" )
else
    ckpt="$em/best.ckpt"
    if [ ! -f "$ckpt" ]; then
        echo "Model file '$ckpt' doesn't exist. Did you set -$MDL_FLG," \
            "-$VER_FLG correctly?"
        exit 1
    fi
    superb_flags=( "-u" "scpc_local" "-k" "$ckpt" "-g" "$expert_config" )
fi

if [ -z "$libri" ]; then
    libri="$dl/local/data"
    if [ ! -f "$libri/.100_complete" ]; then
        echo "Downloading librispeech"
        $cmd_p python prep/librispeech.py "$dl" download ${TR2DL_ARGS[100]}
        touch "$libri/.100_complete"
        ((only)) && exit 0
    fi
fi

if [ ! -f "$sp/config.yaml" ]; then
    echo "Writing config"
    mkdir -p "$sp"
    {
        export libri nwork sp
        cat "conf/superb.$stask.config.template.yaml" | envsubst
    } > "$sp/config.yaml"
    ((only)) && exit 0
fi

if [ "$stask" = "asr" ]; then
    mkdir -p "$sp/len_for_bucket"
    for sasrpart in train-clean-100 dev-clean test-clean; do
        if [ ! -f "$sp/len_for_bucket/$sasrpart.csv" ]; then
            echo "Making $sasrpart length buckets"
            echo "${SASRPART2IDX[$sasrpart]}" |
            python s3prl/s3prl/preprocess/generate_len_for_bucket.py \
                -i "$libri" -o "$sp" -a .flac --n_jobs $nwork
            ((only)) && exit 0
        fi
    done
fi


if [ ! -f "$sp/.train_complete" ]; then
    echo "Training for SUPERB task $stask using model $md"
    $cmd python s3prl/s3prl/run_downstream.py \
        -p "$sp" \
        -c "$sp/config.yaml" \
        "${superb_flags[@]}" \
        -m train \
        -d $darg \
        -a
    touch "$sp/.train_complete"
    ((only)) && exit 0
fi

if [ ! -f "$sp/results.txt" ]; then
    echo "Evaluating for SUPERB task $stask using model $md"
    $cmd python s3prl/s3prl/run_downstream.py \
        -m evaluate -e "$sp/dev-best.ckpt" | 
        grep "test per" | tee "$sp/results.txt" || \
        (rm -f "$sp/results.txt" && exit 1)
    find "$sp/" \
        -type f \
        -and -not -name 'dev-best.ckpt' \
        -and -not -name 'config.yaml' \
        -and -not -name '*tfevents*' \
        -and -not -name '.train_complete' \
        -and -not -name 'results.txt' \
        -delete
    ((only)) && exit 0
fi