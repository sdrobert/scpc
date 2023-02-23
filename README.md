# scpc

Experimenting with smaller windows in CPC acoustic modelling

See the [README](resources/README) for more details on the phone alignments
and train/test split.

## Installation

If you're just interested in the package and its direct dependencies, try

``` sh
pip install git+https://github.com/sdrobert/scpc.git
```

If you want to follow the recipe in [run.sh](run.sh), clone this repo, then

``` sh
git submodule update --init  # populates prep and s3prl
conda env create -f conf/run-environment.yaml
conda activate scpc
pip install .
```

Our compatibility layer for [s3prl](https://github.com/s3prl/s3prl) does not
require any dependencies beyond those necessary for the package itself.
However, we do require the use of a [forked version of
s3prl](https://github.com/sdrobert/s3prl/tree/scpc) to [incorporate our new
upstream](https://s3prl.github.io/s3prl/contribute/upstream.html).

If you're following along from the recipe, you can install that fork of s3prl
in the same environment

``` sh
conda env update --name scpc -f conf/s3prl-environment.yaml
pip install ./s3prl
```

Alternatively, if you just want the s3prl stuff, you can install the branch
into a new environment

``` sh
pip install git+https://github.com/sdrobert/s3prl/tree/scpc
```

## License

This repository is Apache 2.0-licensed. See [LICENSE](LICENSE) for the full
license details.

[cpc_audio](https://github.com/facebookresearch/CPC_audio), which this
repository borrows from, is MIT-licensed. See
[LICENSE_cpc_audio](LICENSE_cpc_audio).
