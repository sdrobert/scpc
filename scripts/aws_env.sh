#! /usr/bin/env

source activate pytorch
conda update --name pytorch -f environment.yaml
# conda activate pytorch
# conda install tensorboard pytorch-lightning=1.
# conda install -c coml virtual-dataset zerospeech-benchmarks zerospeech-libriabx2 zerospeech-tde
# conda install -c sdrobert pydrobert-kaldi pydrobert-param
# pip install "git+https://github.com/sdrobert/pydrobert-pytorch.git@scpc"
# pip install '.'
