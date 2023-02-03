#! /usr/bin/env bash

python prep/librispeech.py data/librispeech download
python prep/librispeech.py data/librispeech preamble --speakers-are-readers
python prep/librispeech.py data/librispeech init_word
python prep/librispeech.py data/librispeech torch_dir wrd raw_100 --compute-up-to 100 --raw

chunk-torch-spect-data-dir \
    data/librispeech/raw_100/{train_clean_100,train_clean_100_chunked20480} \
    --window-type=causal \
    --lobe-size=20479