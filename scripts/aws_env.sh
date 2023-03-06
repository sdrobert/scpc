#! /usr/bin/env

source activate pytorch
conda activate pytorch
conda install tensorboard
conda install -c coml virtual-dataset zerospeech-benchmarks zerospeech-libriabx2 zerospeech-tde
conda install -c sdrobert pydrobert-kaldi pydrobert-param
pip install "git+https://github.com/sdrobert/pydrobert-pytorch.git@scpc"
pip install '.'
