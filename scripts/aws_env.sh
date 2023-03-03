#! /usr/bin/env

source /opt/conda/envs/pytorch/bin/activate
conda install -c coml virtual-dataset zerospeech-benchmarks zerospeech-libriabx2 zerospeech-tde
conda install -c sdrobert pydrobert-kaldi pydrobert-param
pip install "git+https://github.com/sdrobert/pydrobert-pytorch.git@scpc"
pip install '.'
