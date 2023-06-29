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
bl="$em/baseline"
ckpt2="$bl/best.pt"
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
        --custom-lm-max-order 10 \
        --custom-lm-prune-counts 0 0 0 0 0 1 1 1 2 3
    touch "$dl/.960_complete2"
    ((only)) && exit 0
fi

for x in train_clean_100 dev_clean dev_other test_clean test_other; do
  pdir="$dlf/$x"
  if [ ! -f "$pdir/.a2r_complete" ]; then
    echo "Computing representations in $pdir"
    tmp="$em/tmp/$x"
    mkdir -p "$tmp" "$pdir/feat"
    cat "$dl/local/char/$x.wav.scp" | xargs -P $nwork -I {} bash -c '
ln -sf "${1#* }" "${1%% *}.flac" ' -- "$tmp/"{}
    $cmd_p scpc-a2r $expert_args --audio-suffix .flac \
        "$tmp" "$ckpt" "$pdir/feat"
    touch "$pdir/.a2r_complete"
    ((only)) && exit 0
  fi
  rm -rf "$em/tmp"
done

if [ ! -f "$dlf/.100_complete" ]; then
    echo "Converting into SpectDataSet"
    ln -sf "$(cd "$dl/local"; pwd -P)" "$dlf/../local"
    $cmd_p python prep/librispeech.py \
        "$dlf/.." torch_dir char reps --feats-from reps ${TR2TD_ARGS[100]}
    $cmd_p prep/arpa-lm-to-state-dict.py \
        --save-sos \
        "$dlf/ext/"{arpa.lm.gz,token2id.txt,lm.pt}
    $cmd_p compute-mvn-stats-for-torch-feat-data-dir \
        --num-workers $nwork \
        "$dlf/train_clean_100/feat" "$dlf/ext/mvn.pt"
    rm -f "$dlf/../local"
    touch "$dlf/.100_complete"
    ((only)) && exit 0
fi

for x in model data training; do
    f="$bl/$x.yaml"
    if [ ! -f "$f" ]; then
        echo "Writing $f"
        mkdir -p "$bl"
        export input_size="$(scpc-info $expert_args "$ckpt" | awk '$1 == "output_size" {print $2}')"
        export vocab_size="$(awk -v v=0 '{if ($1 > v) v=$1} END {print v + 1}' "$dlf/ext/id2token.txt")"
        export eos="$(awk '$2 == "</s>" {print $1}' "$dlf/ext/id2token.txt")"
        cat "conf/baseline.$x.template.yaml" | envsubst > "$f"
        ((only)) && exit 0
    fi
done

if [ ! -f "$ckpt2" ]; then
    echo "Training baseline for $model"
    if [ $nproc -gt 1 ]; then
        train_script="torchrun --standalone --nproc_per_node=$nproc"
        xtra_args="--distributed-backend nccl $x"
    else
        train_script=""
    fi
    state_dir="$bl/states"
    mkdir -p "$state_dir"
    $cmd $train_script prep/asr_baseline.py \
        --read-model-yaml "$bl/model.yaml" \
        --num-workers $nwork \
        --mvn-path "$dlf/ext/mvn.pt" \
        train \
            --read-data-yaml "$bl/data.yaml" \
            --read-training-yaml "$bl/training.yaml" \
            --state-dir "$state_dir" \
            --state-csv "$bl/training.csv" \
            $xtra_args "$dlf/"{train_clean_100,dev_clean} "$ckpt2"
    ((only)) && exit 0
fi
