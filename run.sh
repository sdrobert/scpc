#! /usr/bin/env bash

# Download librispeech and turn into (raw) tensors. train-clean-100 only train
python prep/librispeech.py data/librispeech download
python prep/librispeech.py data/librispeech preamble \
    --speakers-are-readers --exclude-subsets
python prep/librispeech.py data/librispeech init_word
python prep/librispeech.py data/librispeech torch_dir wrd raw_100 \
    --compute-up-to 100 --raw

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
            data/librispeech/raw_100/train_clean_100/ali &
done
wait
