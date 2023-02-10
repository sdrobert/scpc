#! /usr/bin/env bash

# Download librispeech and turn into (raw) tensors.
# python prep/librispeech.py data/librispeech download
python prep/librispeech.py data/librispeech preamble \
    --speakers-are-readers --exclude-subsets
python prep/librispeech.py data/librispeech init_word
python prep/librispeech.py data/librispeech torch_dir wrd raw --raw

# convert 10ms frame alignments to sample alignments and store the results in
# train_clean_100/ali
#
# N.B. (16000 samps / sec) / (100 frames / sec) = 160 samps / frame
I=10  # number of parallel processes. You can adjust to your system
for i in $(seq 1 $I); do
    unzip -cq resources/converted_aligned_phones.zip | \
        awk -v spf=160 -v i=$i -v I=$I '
    (NR + i - 2) % I == 0 {
        printf "lbi-%s", $1;
        for (n=2; n <= NF; ++n) for (m=0; m < spf; ++m) printf " %s", $n;
        printf "\n";
    }' | \
        write-table-to-torch-dir \
            -i iv -o long \
            'ark,t,s,o:-' \
            data/librispeech/raw/train_clean_100/ali &
done
wait

for x in train test; do
    subset-torch-spect-data-dir \
        data/librispeech/raw/train_clean_100{,_${x}_subset}
        --utt-list-file resources/train_clean_100_${x}_subset.txt
done

# scpc \
#     --read-data-yaml conf/data.libri.raw_100.yaml \
#     fit exp/libri.raw100.cpc.20480.ckpt \
#         --read-model-yaml conf/model.cpc.20480.yaml \
#         @conf/trainer.args.txt

# scpc --read-data-yaml conf/data.libri.raw_100.yaml predict exp/libri.raw100.cpc.20480.ckpt exp/out