#! /usr/bin/env bash

data="${1:-data/librispeech}"
exp="${2:-exp}"
LIBRISPEECH_DIR="$3"
I=10

set -e

# Download librispeech and turn into (raw) tensors.
if [ -z "$LIBRISPEECH_DIR" ]; then
    LIBRISPEECH_DIR="$data/local/data"
    if [ ! -f "$LIBRISPEECH_DIR/.complete" ]; then
        python prep/librispeech.py "$data" download
        touch "$LIBRISPEECH_DIR/.complete"
    fi
fi
if [ ! -f "$data/.complete.1" ]; then
    python prep/librispeech.py data/librispeech preamble \
        --speakers-are-readers --exclude-subsets "$LIBRISPEECH_DIR"
    python prep/librispeech.py data/librispeech init_word "$LIBRISPEECH_DIR"
    touch "$data/.complete.1"
fi
if [ ! -f "$data/.complete.2" ]; then
    python prep/librispeech.py data/librispeech torch_dir wrd raw --raw
    touch "$data/.complete.2"
fi
if [ ! -f "$data/.complete.3" ]; then
    # convert 10ms frame alignments to sample alignments and store the results in
    # train_clean_100/ali
    #
    # N.B. --snip-edges=true for kaldi, which means
    #   frames = (samps - 400) // 160 + 1, or
    #   samps <= 160 * (frames - 1) + 400
    #
    # since the first frame starts at sample 0, the extra 240 to 399 samples are
    # at the end of the recording. We thus use the alignment of the final frame
    # for an additional 399 frames, then crop using get-torch-spect-data-dir-info.
    for i in $(seq 1 $I); do
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
                'ark,t,s,o:-' \
                "$data/raw/train_clean_100/ali" &
    done
    wait
    touch "$data/.complete.3"
fi

if [ ! -f "$data/raw/ext/train_clean_100.info.ark" ]; then
    get-torch-spect-data-dir-info --fix 399 \
        "$data/raw/"{train_clean_100,ext/train_clean_100.info.ark} || \
    rm -f "$data/raw/ext/train_clean_100.info.ark"
fi

if [ ! -f "$data/.complete.4" ]; then
    for x in train test; do
        subset-torch-spect-data-dir \
            "$data/raw/train_clean_100"{,_${x}_subset} \
            --utt-list-file resources/train_clean_100_${x}_subset.txt
    done
    touch "$data/.complete.4"
fi

if [ ! -f "$exp/cpc.20480/version_0/best.ckpt" ]; then
    scpc \
        fit \
            --read-data-yaml conf/data.libri.raw.yaml \
            --read-model-yaml conf/model.cpc.20480.yaml \
            @conf/trainer.args.txt \
            "--train-dir=$data/raw/train_clean_100_train_subset" \
            "--val-dir=$data/raw/train_clean_100_test_subset" \
            "--default_root_dir=$exp"
fi
