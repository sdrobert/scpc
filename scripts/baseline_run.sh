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
blt="$bl/tuning"
bld="$bl/decoding"
ckpt_2kshort="$bl/2kshort.pt"
ckpt_final="$bl/final.pt"
# XXX(sdrobert): we save to experiment dir because reps are specific to the
# experiment
dlf="$em/reps"

WIDTHS=( 1 4 16 64 )
BETAS=( 0 1 )

ckpt_pre="$em/best.ckpt"
if [ ! -f "$ckpt_pre" ]; then
    echo "'$ckpt_pre' is not a file (did you finish ./run.sh?)"
    exit 1
fi

# why the 960hr set if we're only training on clean 100? To ensure we have
# access to the transcripts of all partitions for LM training
if [ -z "$libri" ] && [ ! -f "$dl/.960_complete" ]; then
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
if [ ! -f "$dl/.bl_complete" ]; then
    echo "Performing common prep of librispeech"
    $cmd_p python prep/librispeech.py "$dl" preamble \
        --speakers-are-readers "$libri"
    $cmd_p python prep/librispeech.py "$dl" init_char "$libri" \
        --custom-lm-max-order 10 \
        --custom-lm-prune-counts 0 0 0 0 0 1 1 1 2 3
    touch "$dl/.bl_complete"
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
        "$tmp" "$ckpt_pre" "$pdir/feat"
    touch "$pdir/.a2r_complete"
    ((only)) && exit 0
  fi
  rm -rf "$em/tmp"
done

if [ ! -f "$dlf/.complete" ]; then
    echo "Converting into SpectDataSet"
    ln -sf "$(cd "$dl/local"; pwd -P)" "$dlf/../local"
    $cmd_p python prep/librispeech.py \
        "$dlf/.." torch_dir char reps --feats-from reps ${TR2TD_ARGS[100]}
    $cmd_p prep/arpa-lm-to-state-dict.py \
        --save-sos \
        "$dlf/ext/"{lm.arpa.gz,token2id.txt,lm.pt}
    $cmd_p compute-mvn-stats-for-torch-feat-data-dir \
        --num-workers $nwork \
        "$dlf/train_clean_100/feat" "$dlf/ext/mvn.pt"
    rm -f "$dlf/../local"
    touch "$dlf/.complete"
    ((only)) && exit 0
fi

for x in model data 2kshort-training training; do
    f="$bl/$x.yaml"
    if [ ! -f "$f" ]; then
        echo "Writing $f"
        mkdir -p "$bl"
        export delta_order=0
        export input_size="$(scpc-info $expert_args "$ckpt_pre" | awk -v d=$delta_order '$1 == "output_size" {print $2 * (1 + d)}')"
        export vocab_size="$(awk -v v=0 '{if ($1 > v) v=$1} END {print v + 1}' "$dlf/ext/id2token.txt")"
        export eos="$(awk '$2 == "</s>" {print $1}' "$dlf/ext/id2token.txt")"
        cat "conf/baseline.$x.template.yaml" | envsubst > "$f"
        ((only)) && exit 0
    fi
done

if [ ! -f "$ckpt_2kshort" ]; then
    echo "Training baseline for $model with 2k shortest utterances"
    if [ $nproc -gt 1 ]; then
        train_script="torchrun --standalone --nproc_per_node=$nproc"
        xtra_args="--distributed-backend nccl $x"
    else
        train_script=""
    fi
    state_dir="$bl/states_2kfinal"
    mkdir -p "$state_dir"
    $cmd $train_script prep/asr_baseline.py \
        --read-model-yaml "$bl/model.yaml" \
        --mvn-path "$dlf/ext/mvn.pt" \
        train \
            --num-workers $nwork \
            --read-data-yaml "$bl/data.yaml" \
            --read-training-yaml "$bl/2kshort-training.yaml" \
            --state-dir "$state_dir" \
            --state-csv "$bl/2kshort-training.csv" \
            $xtra_args "$dlf/"{train_2kshort,dev_clean} "$ckpt_2kshort"
    rm -rf "$state_dir"
    ((only)) && exit 0
fi

if [ ! -f "$ckpt_final" ]; then
    echo "Training baseline for $model"
    if [ $nproc -gt 1 ]; then
        train_script="torchrun --standalone --nproc_per_node=$nproc"
        xtra_args="--distributed-backend nccl $x"
    else
        train_script=""
    fi
    state_dir="$bl/states_final"
    mkdir -p "$state_dir"
    $cmd $train_script prep/asr_baseline.py \
        --read-model-yaml "$bl/model.yaml" \
        --mvn-path "$dlf/ext/mvn.pt" \
        train \
            --init-state-dict "$ckpt_2kshort" \
            --num-workers $nwork \
            --read-data-yaml "$bl/data.yaml" \
            --read-training-yaml "$bl/training.yaml" \
            --state-dir "$state_dir" \
            --state-csv "$bl/training.csv" \
            $xtra_args "$dlf/"{train_clean_100,dev_clean} "$ckpt_final"
    rm -rf "$state_dir"
    ((only)) && exit 0
fi

rem=$nproc
for width in "${WIDTHS[@]}"; do
    for beta in "${BETAS[@]}"; do
        Tdir="$blt/${width}_${beta}"
        if [ ! -f "$Tdir/.complete" ]; then
            mkdir -p "$Tdir"
            echo "Checking combination --beam-width $width --beta $beta"
            cat << EOF > "$Tdir/decode.args.txt"
--beam-width
$width
--beta
$beta
EOF
            $cmd_p prep/asr_baseline.py \
                --read-model-yaml "$bl/model.yaml" \
                --mvn-path "$dlf/ext/mvn.pt" \
                decode \
                    "@$Tdir/decode.args.txt" \
                    --lookup-lm-state-dict "$dlf/ext/lm.pt" \
                    --read-data-yaml "$bl/data.yaml" \
                    --max-hyp-len 500 \
                    "$ckpt_final" "$dlf/dev_clean" $Tdir && \
                touch "$Tdir/.complete" &
            rem=$(($rem - 1))
            if [ $rem = 0 ]; then
                declare -i err=0 werr=0
                while wait -n || werr=$?; ((werr != 127)); do
                    err=$werr
                done
                if [ $err -ne 0 ]; then
                    echo "Decoding failed!"
                    exit 1
                fi
                ((only)) && exit 0
                rem=$nproc
            fi
        fi
    done
done

for width in "${WIDTHS[@]}"; do
    for beta in "${BETAS[@]}"; do
        Tdir="$blt/${width}_${beta}"
        if [ ! -f "$Tdir/score.txt" ]; then
            $cmd_p compute-torch-token-data-dir-error-rates \
                --quiet \
                "$dlf/dev_clean/ref" "$Tdir"  > "$Tdir/score.txt" \
                || (rm -f "$Tdir/score.txt" && exit 1)
            rm -f "$Tdir/"*.pt
            ((only)) && exit 0
        fi
    done
done

if [ ! -f "$bl/lm-tuned.decode.args.txt" ]; then
    best_er=1000
    best_args=DEADBEEF
    for beta in "${BETAS[@]}"; do
        [ $beta -eq 0 ] && continue
        for width in "${WIDTHS[@]}"; do
            Tdir="$blt/${width}_${beta}"
            er=$(cat "$Tdir/score.txt")
            if (( $(echo "$er < $best_er" | bc -l) )); then
                best_er=$er
                best_args="$Tdir/decode.args.txt"
            fi
        done
    done
    cp "$best_args" "$bl/lm-tuned.decode.args.txt"
    ((only)) && exit 0
fi

if [ ! -f "$bl/nolm-tuned.decode.args.txt" ]; then
    best_er=1000
    best_args=DEADBEEF
    for width in "${WIDTHS[@]}"; do
        Tdir="$blt/${width}_0"
        er=$(cat "$Tdir/score.txt")
        if (( $(echo "$er < $best_er" | bc -l) )); then
            best_er=$er
            best_args="$Tdir/decode.args.txt"
        fi
    done
    cp "$best_args" "$bl/nolm-tuned.decode.args.txt"
    ((only)) && exit 0
fi

rem=$nproc
for x in dev_clean dev_other test_clean test_other; do
    src="$dlf/$x"
    for y in lm nolm; do
        dst="$bld/$x/$y"
        if [ ! -f "$dst/.complete" ]; then
            mkdir -p "$dst"
            echo "Decoding $x with $y"
            $cmd_p prep/asr_baseline.py \
                --read-model-yaml "$bl/model.yaml" \
                --mvn-path "$dlf/ext/mvn.pt" \
                decode \
                    "@$bl/$y-tuned.decode.args.txt" \
                    --lookup-lm-state-dict "$dlf/ext/lm.pt" \
                    --read-data-yaml "$bl/data.yaml" \
                    --max-hyp-len 500 \
                    "$ckpt_final" "$src" "$dst" && \
                touch "$dst/.complete" &
            rem=$(($rem - 1))
            if [ $rem = 0 ]; then
                declare -i err=0 werr=0
                while wait -n || werr=$?; ((werr != 127)); do
                    err=$werr
                done
                if [ $err -ne 0 ]; then
                    echo "Decoding failed!"
                    exit 1
                fi
                ((only)) && exit 0
                rem=$nproc
            fi
        fi
    done
done

for x in dev_clean dev_other test_clean test_other; do
    echo "Computing final error rates for $x..."
    if [ ! -f "$bld/scores.$x.txt" ]; then
        cp -f "$dlf/ext/$x.ref.trn" "$bld/$x.ref.chr.trn"
        $cmd_p prep/subword2word.py "$bld/$x.ref."{chr,wrd}".trn"
        for y in lm nolm; do
            $cmd_p torch-token-data-dir-to-trn \
                --num-workers $nwork \
                "$bld/$x/$y" "$dlf/ext/id2token.txt" "$bld/$x.hyp.chr.$y.trn"
            $cmd_p prep/subword2word.py "$bld/$x.hyp."{chr,wrd}".$y.trn"
        done
        for y in chr wrd; do
            prep/error-rates-from-trn.py --suppress-warning \
                "$bld/$x.ref.$y.trn" "$bld/$x.hyp.$y."{lm,nolm}.trn \
                > "$bld/scores.$x.$y.txt"
        done
        cat "$bld/scores.$x."*.txt > "$bld/scores.$x.txt"
        ((only)) && exit 0
    fi
done

cat "$bld/scores."{dev_clean,dev_other,test_clean,test_other}.txt
