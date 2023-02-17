#! /usr/bin/env bash

data="${1:-data}"
exp="${2:-exp}"
model="${3:-cpc.20480}"
LIBRISPEECH_DIR="$3"

I=1
dl="$data/librispeech"
dz="$data/zerospeech"
pdl="$exp/$model/predict/librispeech"
zs="$exp/$model/zrc"
export APP_DIR="$dz/local/data"

set -e

if [ -z "$LIBRISPEECH_DIR" ] &&  [ ! -f "$dl/.complete.2" ]; then
    # Download librispeech and turn into (raw) tensors.
    LIBRISPEECH_DIR="$dl/local/data"
    if [ ! -f "$LIBRISPEECH_DIR/.complete" ]; then
        python prep/librispeech.py "$dl" download
        touch "$LIBRISPEECH_DIR/.complete"
    fi
fi
if [ ! -f "$dl/.complete.1" ]; then
    python prep/librispeech.py data/librispeech preamble \
        --speakers-are-readers --exclude-subsets "$LIBRISPEECH_DIR"
    python prep/librispeech.py data/librispeech init_word "$LIBRISPEECH_DIR"
    touch "$dl/.complete.1"
fi
if [ ! -f "$dl/.complete.2" ]; then
    python prep/librispeech.py data/librispeech torch_dir wrd raw --raw
    touch "$dl/.complete.2"
fi
if [ ! -f "$dl/.complete.3" ]; then
    # convert 10ms frame alignments to sample alignments and store the results
    # in train_clean_100/ali
    #
    # N.B. --snip-edges=true for kaldi, which means frames = (samps - 400) //
    #   160 + 1, or samps <= 160 * (frames - 1) + 400
    #
    # since the first frame starts at sample 0, the extra 240 to 399 samples
    # are at the end of the recording. We thus use the alignment of the final
    # frame for an additional 399 frames, then crop using
    # get-torch-spect-data-dir-info.
    for i in $(seq 1 $I); do
        rm -f "$dl/.complete.3-$i"
        unzip -cq resources/converted_aligned_phones.zip | \
            awk -v spf=160 -v pad=399 -v i=$i -v I=$I '
        (NR + i - 2) % I == 0 {
            printf "lbi-%s", $1;
            for (n=2; n <= NF; ++n) for (m=0; m < spf; ++m) printf " %s", $n;
            for (m=0; m < pad; ++m) printf " %s", $NF;
            printf "\n";
        }' | \
            write-table-to-torch-dir \
                -i iv -o long \
                'ark,t,s,o,cs:-' \
                "$dl/raw/train_clean_100/ali" && \
            touch "$dl/.complete.3-$i" &
    done
    wait
    for i in $(seq 1 $I); do
        if [ ! -f "$dl/.complete.3-$i" ]; then
            echo -e "Process $i/$I failed!"
            exit 1
        fi
    done
    touch "$dl/.complete.3"
fi

if [ ! -f "$dl/raw/ext/train_clean_100.info.ark" ]; then
    # primarily used to fix the alignments which pad too many samples
    get-torch-spect-data-dir-info --fix 399 \
        "$dl/raw/"{train_clean_100,ext/train_clean_100.info.ark} || \
    rm -f "$dl/raw/ext/train_clean_100.info.ark"
fi

if [ ! -f "$dl/.complete.4" ]; then
    for x in train test; do
        subset-torch-spect-data-dir \
            "$dl/raw/train_clean_100"{,_${x}_subset} \
            --utt-list-file resources/train_clean_100_${x}_subset.txt
    done
    touch "$dl/.complete.4"
fi

# This step doesn't need to be performed usually. The goal is to map
# librispeech predictions to their zerospeech submission files. re-downloading
# parts of librispeech is unnecessary
# if [ ! -f "$dz/.complete" ]; then
#     zrc datasets:pull abxLS-dataset
#     find "$APP_DIR/datasets/abxLS-dataset" -name '*.wav' | \
#         awk -F "/" '
# {
#     NFm1=NF-1;
#     dn=$NFm1;
#     gsub(/-/, "_", dn);
#     split($NF, fn, ".");
#     print dn"/lbi-"fn[1]".npy "$NFm1"/"fn[1]".npy"
# }' > resources/libri_to_abxLS.map
#     touch "$dz/.complete"
# fi

if [ ! -f "$exp/$model/version_0/best.pt" ]; then
    scpc \
        --read-model-yaml "conf/model.$model.yaml" \
        fit \
            --read-data-yaml conf/data.libri.raw.yaml \
            @conf/trainer.args.txt \
            "--train-dir=$dl/raw/train_clean_100_train_subset" \
            "--val-dir=$dl/raw/train_clean_100_test_subset" \
            "--default_root_dir=$exp"
fi

for x in dev_clean dev_other test_clean test_other; do
    # compute predictions
    if [ ! -f "$pdl/$x/.complete" ]; then
        mkdir -p "$pdl/$x"
        scpc \
            --read-model-yaml "conf/model.$model.yaml" \
            predict --numpy --device=cuda \
            "$exp/$model/version_0/best.pt" "$dl/raw/$x" "$pdl/$x"
        touch "$pdl/$x/.complete"
    fi
done

if [ ! -f "$zs/.complete.1" ]; then
    # set up zerospeech abxLS submission
    rm -rf "$zs"
    zrc submission:init abxLS "$zs"
    scpc --read-model-yaml "conf/model.$model.yaml" info | \
        awk '
BEGIN {spf=0}
NR == FNR && $1 == "downsampling_factor" {spf=$2 / 16000}
NR != FNR {if ($1 == "feature_size:") $2=spf; print}' \
            - conf/params.template.yaml > "$zs/params.yaml"
    awk '
BEGIN {sd="\"my great model\""}
NR == FNR && $1 == "system_description:" {$1=""; split($0, x, "#"); sd=x[1]}
NR != FNR {if ($1 == "system_description:") $2=sd; print}
' conf/{model.$model.yaml,meta.template.yaml} > "$zs/meta.yaml"
    awk -v "di=$pdl/" -v "do_=$zs/" '{$1="\""di$1"\""; $2="\""do_$2"\""; print}' \
        resources/libri_to_abxLS.map | xargs -P $I -I{} sh -c 'cp -f {}'
    touch "$zs/.complete.1"
fi

if [ ! -f "$zs/.complete.2" ]; then
    # score zerospeech abxLS submission
    zrc benchmarks:run abxLS "$zs"
    touch "$zs/.complete.2"
fi