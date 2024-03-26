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

set -eo pipefail

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

if [ -z "$pca" ]; then
  bl="$em/baseline/full_v${vocab_size}"
  dlf="$em/reps/full"
else
  bl="$em/baseline/pca${pca}_v${vocab_size}"
  dlf="$em/reps/pca${pca}_v${vocab_size}"
fi
bld="$bl/decoding"

ckpt_2kshort="$bl/2kshort.pt"
ckpt_final="$bl/final.pt"
local_sw="$dl/local/sw${vocab_size}"
lm_train_gz="${local_sw}/librispeech-lm-norm-subword.txt.gz"
if [ "$lm_ord" = 0 ]; then
    dname="nolm_width${width}"
    lm_gz=
else
    dname="lm_ord${lm_ord}_width${width}"
    lm_gz="${local_sw}/lm.$lm_ord.arpa.gz"
fi
blt="$bld/tuning/$dname"

ckpt_pre="$em/best.ckpt"
if [ ! -f "$ckpt_pre" ]; then
    echo "'$ckpt_pre' is not a file (did you finish ./run.sh?)"
    exit 1
fi

if [ -z "$libri" ]; then
    libri="$dl/local/data"
    if [ ! -f "$libri/.bl_complete" ] && [ ! -f "$dl/.sw${vocab_size}_complete" ]; then
        echo "Downloading librispeech"
        $cmd_p python prep/librispeech.py "$dl" download ${TR2DL_ARGS[100]}
        [ "$lm_ord" = 0 ] || $cmd_p python prep/librispeech.py "$dl" download \
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
        ln -sf "$x.ref.trn" "${local_sw}/$x.ref.sw${vocab_size}.trn"
        $cmd_p prep/subword2word.py "${local_sw}/$x.ref"{,.wrd}".trn"
    done
    cp -f "$(find "$libri" -name 'librispeech-vocab.txt')" "${local_sw}/"
    touch "$dl/.sw${vocab_size}_complete"
    ((only)) && exit 0
fi

# if ((lm_ord > 0)) && [ ! -f "$lm_train_gz" ]; then
#     echo "Constructing subword LM training file"
#     [ -z "$libri" ] && libri="$dl/local/data"
#     tdata="$(find "$libri" -name "librispeech-lm-norm.txt" | head -n 1)"
#     if [ -z "$tdata" ]; then
#         tdata="$(find "$libri" -name "librispeech-lm-norm.txt.gz" | head -n 1)"
#         if [ -z "$tdata" ]; then
#             echo "Could not find librispeech-lm-norm.txt[.gz] in '$libri'"
#             exit 1
#         fi
#         tdata_cmd=( gunzip -c "$tdata" )
#     else
#         tdata_cmd=( cat "$tdata" )
#     fi
#     $cmd_p "${tdata_cmd[@]}" |
#         prep/word2subword.py -s "${local_sw}/spm.model" --both-raw |
#         gzip -c > "${lm_train_gz}_"
#     gzip -t "${lm_train_gz}_" || (rm -f "${lm_train_gz}_" && exit 1)
#     mv "${lm_train_gz}"{_,}
#     ((only)) && exit 0
# fi

if ((lm_ord > 0)) && [ ! -f "$lm_gz" ]; then
    echo "Building $lm_ord-gram word-level language model"
    [ -z "$libri" ] && libri="$dl/local/data"
    if which lmplz 2>&1 > /dev/null; then
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
    tdata="$(find "$libri" -name "librispeech-lm-norm.txt" | head -n 1)"
    if [ -z "$tdata" ]; then
        tdata="$(find "$libri" -name "librispeech-lm-norm.txt.gz" | head -n 1)"
        if [ -z "$tdata" ]; then
            echo "Could not find librispeech-lm-norm.txt[.gz] in '$libri'"
            exit 1
        fi
    fi
    $lm_cmd $prunes -o $lm_ord < "$tdata" |
        gzip -c > "${lm_gz}_" 
    if [ $(gunzip -c "${lm_gz}_" 2> /dev/null | head -n 1 | wc -l) != 1 ]; then
        echo "n-gram lm creation failed!"
        rm -f "${lm_gz}_"
        exit 1
    fi
    mv "${lm_gz}"{_,}
    ((only)) && exit 0
fi

# if ((lm_ord > 0)) && [ ! -f "$lm_pt" ]; then
#     echo "Compiling $lm_ord-gram language model as state dict"
#     $cmd_p prep/arpa-lm-to-state-dict.py \
#         --sos-id -1 --save-sos --on-extra drop -v \
#         "$lm_gz" "${local_sw}/token2id.txt" "$lm_pt" \
#             || (rm -f "$lm_pt" && exit 1)
    
#     # N.B. If you want to evaluate the LM in terms of perplexity, you can use
#     # the lookup-lm-corpus-perplexity.py script, e.g.
#     #
#     # awk '{NF-=1; print}' "${local_sw}/dev_clean.ref.trn" > dev_clean.txt
#     # prep/lookup-lm-corpus-perplexity.py \
#     #   --states-and-token2id "${lm_pt}" "${local_sw}/token2id" \
#     #   --chunk-size 64 \
#     #   dev_clean.txt
#     #
#     # This is not necessary and, moreover, slow

#     ((only)) && exit 0
# fi

for x in dev_clean train_clean_100 dev_other test_clean test_other; do
  pdir="$dlf/$x"
  if [ ! -f "$pdir/.a2r_complete" ] && [ ! -f "$bl/.deepcleaned" ]; then
    tmp="$em/tmp"
    echo "Splitting wav.scp into $tmp"
    rm -rf "$tmp" "$pdir/feat"
    mkdir -p "$tmp" "$pdir/feat"
    wav_num="$(wc -l "${local_sw}/$x.wav.scp"| cut -d ' ' -f 1)"
    lines_per_proc="$(( (wav_num + nproc - 1) / nproc ))"

    split -dl "$lines_per_proc" "${local_sw}/$x.wav.scp" "$tmp/wav.scp"
    split_num="$(cat "$tmp/wav.scp"* | wc -l)"
    if [ "$wav_num" != "$split_num" ]; then
        echo "Expected $wav_num utts in $tmp/wav.scp*; got $split_num"
        exit 1
    fi

    echo "Computing representations in $pdir"
    for f in "$tmp/wav.scp"*; do
        $cmd_p scpc-a2r $expert_args \
            --in-map "$f" "$ckpt_pre" "$pdir/feat" && \
            touch "$tmp/.complete.$(basename "$f")" &
    done
    
    wait
    for f in "$tmp/wav.scp"*; do
        if [ ! -f "$tmp/.complete.$(basename "$f")" ]; then
            echo "${f##*scp} failed!"
            exit 1
        fi
    done

    touch "$pdir/.a2r_complete"
    rm -rf "$tmp"
    ((only)) && exit 0
  fi
done

if [ ! -f "$dlf/.complete" ] && [ ! -f "$bl/.deepcleaned" ]; then
    echo "Converting into SpectDataSet"
    ln -sf "$(cd "$dl/local"; pwd -P)" "$dlf/../local"
    $cmd_p python prep/librispeech.py \
        "$dlf/.." torch_dir sw${vocab_size} $(basename "$dlf") \
            --skip-verify \
            --feats-from $(basename "$dlf") ${TR2TD_ARGS[100]}
    $cmd_p compute-mvn-stats-for-torch-feat-data-dir \
        --num-workers $nwork \
        "$dlf/train_clean_100/feat" "$dlf/ext/mvn.pt"
    rm -f "$dlf/../local"
    touch "$dlf/.complete"
    ((only)) && exit 0
fi

for x in id2token.txt token2id.txt librispeech-vocab.txt; do
    if [ ! -f "$bl/$x" ]; then
        src="$(find "$local_sw" -name "$x" | head -n 1)"
        if [ ! -f "$src" ]; then
            echo "Could not find $x in $local_sw"
            exit 1
        fi
        mkdir -p "$bl"
        cp "$src" "$bl/"
        ((only)) && exit 0
    fi
done

for x in model data 2kshort-training training; do
    f="$bl/$x.yaml"
    if [ ! -f "$f" ]; then
        echo "Writing $f"
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
        xtra_args_="--distributed-backend nccl $xtra_args"
    else
        train_script=""
        xtra_args_="$xtra_args"
    fi
    state_dir="$bl/states_2kfinal"
    mkdir -p "$state_dir"
    $cmd $train_script prep/asr-baseline.py \
        --read-model-yaml "$bl/model.yaml" \
        --mvn-path "$dlf/ext/mvn.pt" \
        train \
            --num-workers $nwork \
            --read-data-yaml "$bl/data.yaml" \
            --read-training-yaml "$bl/2kshort-training.yaml" \
            --state-dir "$state_dir" \
            --state-csv "$bl/2kshort-training.csv" \
            ${xtra_args_} "$dlf/"{train_2kshort,dev_clean} "$ckpt_2kshort"
    $clean && rm -rf "$state_dir"
    ((only)) && exit 0
fi

if [ ! -f "$ckpt_final" ]; then
    echo "Training baseline for $model"
    if [ $nproc -gt 1 ]; then
        train_script="torchrun --standalone --nproc_per_node=$nproc"
        xtra_args_="--distributed-backend nccl $xtra_args"
    else
        xtra_args_="$xtra_args"
        train_script=""
    fi
    state_dir="$bl/states_final"
    mkdir -p "$state_dir"
    $cmd $train_script prep/asr-baseline.py \
        --read-model-yaml "$bl/model.yaml" \
        --mvn-path "$dlf/ext/mvn.pt" \
        train \
            --init-state-dict "$ckpt_2kshort" \
            --num-workers $nwork \
            --read-data-yaml "$bl/data.yaml" \
            --read-training-yaml "$bl/training.yaml" \
            --state-dir "$state_dir" \
            --state-csv "$bl/training.csv" \
            ${xtra_args_} "$dlf/"{train_clean_100,dev_clean} "$ckpt_final"
    if $clean; then
        rm -rf "$state_dir"
        for x in "$dlf/train_"*; do
            # training data reps are no longer needed
            find "$x" -name 'lbi-*.pt' -delete
        done
    fi
    ((only)) && exit 0
fi

for x in dev_clean dev_other test_clean test_other; do
    src="$dlf/$x"
    logits="$bld/$x/logits"
    if [ ! -f "$logits/.complete" ] && [ ! -f "$bl/.deepcleaned" ]; then
        mkdir -p "$logits"
        echo "Saving logits for $x"
        $cmd_p prep/asr-baseline.py \
            --read-model-yaml "$bl/model.yaml" \
            --mvn-path "$dlf/ext/mvn.pt" \
            decode \
                --read-data-yaml "$bl/data.yaml" \
                --max-hyp-len 500 \
                --write-logits \
                "$ckpt_final" "$src" "$logits"
        touch "$logits/.complete"
        if $clean; then
            # partition's reps are no longer needed
            find "$src" -name 'lbi-*.pt' -delete
        fi
        ((only)) && exit 0
    fi
done

if ((lm_ord)); then
    if [ ! -f "$bld/lm.$lm_ord.gz" ]; then
        echo "Hard-linking lm into $bld"
        ln "$lm_gz" "$bld/lm.$lm_ord.gz"
        ((only)) && exit 0
    fi
fi

if [ "$lm_ord" = 0 ]; then
    AINVS=( 1 )
else
    AINVS=( 1 2 3 4 )
fi
BETAS=( 0 1 2 3 )
for ainv in "${AINVS[@]}"; do
    for beta in "${BETAS[@]}"; do
        mkdir -p "$blt"
        tuning="$blt/alpha-inv${ainv}_beta${beta}"
        if [ ! -f "$tuning.hyp.trn" ]; then
            echo "Tuning with alpha-inv=$ainv, beta=$beta, and width=$width"
            cat > "$tuning.args.txt" <<EOF
--bpe
--words
$bl/librispeech-vocab.txt
--token2id
$bl/token2id.txt
--alpha-inv
$ainv
--beta
$beta
EOF
            if ((lm_ord)); then
            cat >> "$tuning.args.txt" <<EOF
--lm
$bld/lm.$lm_ord.gz
EOF
            fi
            $cmd_p python prep/logits-to-trn-via-pyctcdecode.py \
                "@$tuning.args.txt" \
                --batch-size $nwork \
                "$bld/dev_clean/logits" "$tuning.hyp.trn_"
            mv "$tuning.hyp.trn"{_,}
            ((only)) && exit 0
        fi
    done
done

if [ ! -f "$blt/best.args.txt" ]; then
    python prep/error-rates-from-trn.py \
        --suppress-warning \
        "${local_sw}/dev_clean.ref.wrd.trn" \
        "$blt/"*.hyp.trn > "$blt/scores.txt"
    best="$(grep -Po $'(?<=best hyp \').+(?=\\.hyp\\.trn)' "$blt/scores.txt").args.txt"
    if [ ! -f "$best" ]; then
        echo "Failed to find best tuned args"
        exit 1
    fi
    cp "$best" "$blt/best.args.txt"
    ((only)) && exit 0
fi

for x in dev_clean dev_other test_clean test_other; do
    trn="$bld/$x/$dname.hyp.trn"
    if [ ! -f "$trn" ]; then
        echo "Creating transcriptions for $x"
        $cmd_p python prep/logits-to-trn-via-pyctcdecode.py \
            "@$blt/best.args.txt" \
            --batch-size $nwork \
            "$bld/$x/logits" "$trn"_
        mv "$trn"{_,}
        ((only)) && exit 0
    fi
done

if $clean; then
    rm -rf "$bl/states_"*
    if ! $deepclean; then
        for x in "$dlf/"{dev_clean,dev_other,test_clean,test_other,train_*} ; do
            find "$x/" -name 'lbi-*.pt' -delete
        done
    fi
fi

for x in dev_clean dev_other test_clean test_other; do
    trn="$bld/$x/$dname.hyp.trn"
    score="$bld/$x/$dname.wer.txt"
    if [ ! -f "$score" ]; then
        prep/error-rates-from-trn.py --suppress-warning \
            "${local_sw}/$x.ref.wrd.trn" \
            "$bld/$x/$dname.hyp.trn" | grep '^hyp ' > "$score"
        ((only)) && exit 0
    fi
done


if $deepclean; then
    touch "$bl/.deepcleaned"
    rm -rf "$dlf"  # remove the entire rep folder
    find "$bld" \
        -mindepth 2 -maxdepth 2 -name 'logits' -type d -exec rm -rf {} \;
fi

echo "-----------------------------------------------"
echo "Word error rates"
for x in dev_clean dev_other test_clean test_other; do
    echo "$x"
    cat "$bld/$x/"*.wer.txt
done
