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
help_message="Train/decode baseline ASR model on pre-trained speech reps"
source scripts/preamble.sh

dl="$data/librispeech"
em="$exp/$model/version_$ver"
bl="$em/baseline"

if [ -z "$pca" ]; then
  bl="$em/baseline/full_v${vocab_size}"
  dlf="$em/reps/full"
else
  bl="$em/baseline/pca${pca}_v${vocab_size}"
  dlf="$em/reps/pca${pca}_v${vocab_size}"
fi

blt="$bl/tuning"
bld="$bl/decoding"
ckpt_2kshort="$bl/2kshort.pt"
ckpt_final="$bl/final.pt"
local_sw="$dl/local/sw${vocab_size}"
lm_train_gz="${local_sw}/librispeech-lm-norm-subword.txt.gz"
lm_gz="${local_sw}/lm.$lm_ord.arpa.gz"
lm_pt="${local_sw}/lm.$lm_ord.pt"
conds=( nolm )
if ((lm_ord > 0)); then
    conds+=( lm.$lm_ord )
fi

# if [ ! -z "${NOLM:+xxx}" ]; then
#     echo "Skipping lm condition"
#     with_lm=false
# elif which lmplz 2> /dev/null; then
#     echo "kenlm found. Will add 'lm' condition"
#     CONDS+=( lm )
#     with_lm=true
# else
#     echo "Cannot find kenlm and env var NOLM was not set. Continuing the"
#     echo "recipe without the LM will require you to modify the"
#     echo "recipe/artifacts later. If you're okay with this, define the NOLM"
#     echo "environment variable and try calling this script again. If you "
#     echo "already finished the part of the script which compiles with kenlm, "
#     echo 
#     exit 1
# fi

ckpt_pre="$em/best.ckpt"
if [ ! -f "$ckpt_pre" ]; then
    echo "'$ckpt_pre' is not a file (did you finish ./run.sh?)"
    exit 1
fi

if [ -z "$libri" ] && [ ! -f "$dl/.sw${vocab_size}_complete" ]; then
    libri="$dl/local/data"
    if [ ! -f "$libri/.bl_complete" ]; then
        echo "Downloading librispeech"
        $cmd_p python prep/librispeech.py "$dl" download ${TR2DL_ARGS[100]}
        $with_lm && $cmd_p python prep/librispeech.py "$dl" download \
                --files librispeech-lm-norm.txt.gz
        touch "$libri/.bl_complete"
        ((only)) && exit 0
    fi
fi

if [ ! -f "$dl/.sw${vocab_size}_complete" ]; then
    echo "Performing common prep of librispeech"
    $cmd_p python prep/librispeech.py "$dl" preamble \
        --speakers-are-readers "$libri"
    $cmd_p python prep/librispeech.py "$dl" init_subword "$libri" \
        --vocab-size ${vocab_size} --config-subdir "sw${vocab_size}"
    for x in dev_clean train_clean_100 dev_other test_clean test_other; do
        ln -sf "${local_sw}/$x.ref"{,.sw${vocab_size}}".trn"
        $cmd_p prep/subword2word.py "${local_sw}/$x.ref"{,.wrd}".trn"
    done
    touch "$dl/.sw${vocab_size}_complete"
    ((only)) && exit 0
fi

if ((lm_ord > 0)) && [ ! -f "$lm_train_gz" ]; then
    echo "Constructing subword LM training file"
    [ -z "$libri" ] && libri="$dl/local/data"
    tdata="$(find "$libri" -name "librispeech-lm-norm.txt" | head -n 1)"
    if [ -z "$tdata" ]; then
        tdata="$(find "$libri" -name "librispeech-lm-norm.txt.gz" | head -n 1)"
        if [ -z "$tdata" ]; then
            echo "Could not find librispeech-lm-norm.txt[.gz] in '$libri'"
            exit 1
        fi
        tdata_cmd=( gunzip -c "$tdata" )
    else
        tdata_cmd=( cat "$tdata" )
    fi
    $cmd_p "${tdata_cmd[@]}" |
        prep/word2subword.py -s "${local_sw}/spm.model" --both-raw |
        gzip -c > "${lm_train_gz}_"
    gzip -t "${lm_train_gz}_" || (rm -f "${lm_train_gz}_" && exit 1)
    mv "${lm_train_gz}"{_,}
    ((only)) && exit 0
fi

if ((lm_ord > 0)) && [ ! -f "$lm_gz" ]; then
    echo "Building $lm_ord-gram subword-level language model"
    [ -z "$libri" ] && libri="$dl/local/data"
    if which lmplz 2> /dev/null; then
        echo "found kenlm (lmplz). Using it to build lm"
        lm_cmd="lmplz -S 8G --discount_fallback 0.5 1.0 1.5 --prune"
    else
        echo "did not find kenlm (lmplz). Using my dumb ngram_lm.py file"
        lm_cmd="prep/ngram_lm.py -T -v -t"
    fi
    if [ $lm_ord = 1 ]; then
        prunes="0"
    else
        prunes="0 1"
    fi
    $cmd_p $lm_cmd $prunes -o $lm_ord < "$lm_train_gz" |
        gzip -c > "${lm_gz}_" 
    if [ $(gunzip -c "${lm_gz}_" 2> /dev/null | head -n 1 | wc -l) != 1 ]; then
        echo "n-gram lm creation failed!"
        rm -f "${lm_gz}_"
        exit 1
    fi
    mv "${lm_gz}"{_,}
    ((only)) && exit 0
fi

if ((lm_ord > 0)) && [ ! -f "$lm_pt" ]; then
    echo "Compiling $lm_ord-gram language model as state dict"
    $cmd_p prep/arpa-lm-to-state-dict.py \
        --sos-id -1 --save-sos --on-extra drop -v \
        "$lm_gz" "${local_sw}/token2id.txt" "$lm_pt" \
            || (rm -f "$lm_pt" && exit 1)
    
    # N.B. If you want to evaluate the LM in terms of perplexity, you can use
    # the lookup-lm-corpus-perplexity.py script, e.g.
    #
    # awk '{NF-=1; print}' "${local_sw}/dev_clean.ref.trn" > dev_clean.txt
    # prep/lookup-lm-corpus-perplexity.py \
    #   --states-and-token2id "${lm_pt}" "${local_sw}/token2id" \
    #   --chunk-size 64 \
    #   dev_clean.txt
    #
    # This is not necessary and, moreover, slow

    ((only)) && exit 0
fi

for x in dev_clean train_clean_100 dev_other test_clean test_other; do
  pdir="$dlf/$x"
  if [ ! -f "$pdir/.a2r_complete" ]; then
    tmp="$em/tmp/$x"
    echo "Linking wav files into $tmp"
    rm -rf "$tmp"
    mkdir -p "$tmp" "$pdir/feat"
    wav_num="$(cat "${local_sw}/$x.wav.scp"| wc -l)"
    lines_per_proc="$(( (wav_num + nproc - 1) / nproc ))"

    split -dl "$lines_per_proc" "${local_sw}/$x.wav.scp" "$tmp/wav.scp"
    split_num="$(cat "$tmp/wav.scp"* | wc -l)"
    if [ "$wav_num" != "$split_num" ]; then
        echo "Expected $wav_num utts in $tmp/wav.scp*; got $split_num"
        exit 1
    fi

    for f in "$tmp/wav.scp"*; do
        stmp="$tmp/${f##*scp}"
        mkdir -p "$stmp"
        $cmd_p awk -v tmp="$stmp" '{print $2,tmp"/"$1".flac"}' "$f" |
            xargs -P $nwork -I % bash -c 'ln -sf $1' -- %
    done
    tmp_num="$(find "$tmp" -name '*.flac' | wc -l)"
    if [ "$wav_num" != "$tmp_num" ]; then
        echo "Expected $wav_num utts in $tmp; got $tmp_num"
        exit 1
    fi

    echo "Computing representations in $pdir"
    for f in "$tmp/wav.scp"*; do
        stmp="$tmp/${f##*scp}"
        $cmd_p scpc-a2r $expert_args --audio-suffix .flac \
            "$stmp" "$ckpt_pre" "$pdir/feat" && touch "$stmp/.complete" &
    done
    
    wait
    for f in "$tmp/wav.scp"*; do
        stmp="$tmp/${f##*scp}"
        if [ ! -f "$stmp/.complete" ]; then
            echo "${f##*scp} failed!"
            exit 1
        fi
    done

    touch "$pdir/.a2r_complete"
    rm -rf "$tmp"
    ((only)) && exit 0
  fi
done

if [ ! -f "$dlf/.complete" ]; then
    echo "Converting into SpectDataSet"
    ln -sf "$(cd "$dl/local"; pwd -P)" "$dlf/../local"
    $cmd_p python prep/librispeech.py \
        "$dlf/.." torch_dir sw${vocab_size} $(basename "$dlf") \
            --feats-from $(basename "$dlf") ${TR2TD_ARGS[100]}
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
        (
            if [ -z "$pca" ]; then
                export input_size="$(scpc-info $expert_args "$ckpt_pre" | awk '$1 == "output_size" {print $2}')"
            else
                export input_size="$pca"
            fi
            export vocab_size
            cat "conf/baseline.$x.template.yaml" | envsubst > "$f"
        )
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
    $clean && rm -rf "$state_dir"
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
    if $clean; then
        rm -rf "$state_dir"
        for x in "$dlf/train_"*; do
            find "$x" -name 'lbi-*.pt' -delete
        done
    fi
    ((only)) && exit 0
fi

rem=$nproc
for width in "${WIDTHS[@]}"; do
    for cond in "${conds[@]}"; do
        Tdir="$blt/${cond}_b${width}"
        if [ ! -f "$Tdir/.complete" ]; then
            mkdir -p "$Tdir"
            echo "Checking $cond with beam width $width"
            if [ "$cond" = nolm ]; then
                cat << EOF > "$Tdir/decode.args.txt"
--beam-width
$width
EOF
            else
                cat << EOF > "$Tdir/decode.args.txt"
--beam-width
$width
--lookup-lm-state-dict
$lm_pt
--beta
0.5
EOF
            fi
            $cmd_p prep/asr_baseline.py \
                --read-model-yaml "$bl/model.yaml" \
                --mvn-path "$dlf/ext/mvn.pt" \
                decode \
                    "@$Tdir/decode.args.txt" \
                    --read-data-yaml "$bl/data.yaml" \
                    --max-hyp-len 500 \
                    "$ckpt_final" "$dlf/dev_clean" $Tdir
            touch "$Tdir/.complete"
            ((only)) && exit 0
        fi
    done
done

for width in "${WIDTHS[@]}"; do
    for cond in "${conds[@]}"; do
        Tdir="$blt/${cond}_b${width}"
        if [ ! -f "$Tdir/score.txt" ]; then
            $cmd_p compute-torch-token-data-dir-error-rates \
                --quiet \
                "$dlf/dev_clean/ref" "$Tdir"  > "$Tdir/score.txt" \
                || (rm -f "$Tdir/score.txt" && exit 1)
            $clean && find "$Tdir/" -name 'lbi-*.pt' -delete
            ((only)) && exit 0
        fi
    done
done

for cond in "${conds[@]}"; do
    if [ ! -f "$bl/$cond-tuned.decode.args.txt" ]; then
        best_er=1000
        best_args=DEADBEEF
        for width in "${WIDTHS[@]}"; do
            Tdir="$blt/${cond}_b${width}"
            er=$(cat "$Tdir/score.txt")
            if (( $(echo "$er < $best_er" | bc -l) )); then
                best_er=$er
                best_args="$Tdir/decode.args.txt"
            fi
        done
        cp "$best_args" "$bl/$cond-tuned.decode.args.txt"
        ((only)) && exit 0
    fi
done

rem=$nproc
for x in dev_clean dev_other test_clean test_other; do
    src="$dlf/$x"
    for y in "${conds[@]}"; do
        dst="$bld/$x/$y"
        if [ ! -f "$dst/.complete" ]; then
            mkdir -p "$dst"
            echo "Decoding $x with $y"
            $cmd_p prep/asr_baseline.py \
                --read-model-yaml "$bl/model.yaml" \
                --mvn-path "$dlf/ext/mvn.pt" \
                decode \
                    "@$bl/$y-tuned.decode.args.txt" \
                    --read-data-yaml "$bl/data.yaml" \
                    --max-hyp-len 500 \
                    "$ckpt_final" "$src" "$dst"
            touch "$dst/.complete"
            ((only)) && exit 0
        fi
    done
done

for x in dev_clean dev_other test_clean test_other; do
    for y in "${conds[@]}"; do
        if [ ! -f "$bld/$x.hyp.wrd.$y.trn" ]; then
            echo "Creating transcriptions of $x with $y"
            $cmd_p torch-token-data-dir-to-trn \
                --num-workers $nwork \
                "$bld/$x/$y" "$dlf/ext/id2token.txt" \
                "$bld/$x.hyp.sw${vocab_size}.$y.trn"
            $cmd_p prep/subword2word.py \
                "$bld/$x.hyp."{sw${vocab_size},wrd}".$y.trn"
            $clean && find "$bld/$x/$y" -name 'lbi-*.pt' -delete
            ((only)) && exit 1
        fi
    done
done

if $clean; then
    rm -rf "$bl/states_"*
    for x in "$dlf" "$blt" "$bld"; do
        find "$x/" -name 'lbi-*.pt' -delete
    done
fi

echo "-----------------------------------------------"
echo "Subword error rates with vocab size $vocab_size"
for x in dev_clean dev_other test_clean test_other; do
    prep/error-rates-from-trn.py --suppress-warning \
        "${sw_local}/$x.ref.sw${vocab_size}.trn" \
        "$bld/$x.hyp.sw${vocab_size}."*.trn
done

echo ""
echo "Word error rates"
for x in dev_clean dev_other test_clean test_other; do
    prep/error-rates-from-trn.py --suppress-warning \
        "${sw_local}/$x.ref.wrd.trn" \
        "$bld/$x.hyp.wrd."*.trn
done
