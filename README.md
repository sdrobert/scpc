# scpc

Package and experiments for pre-training (mostly) CPC speech representations.

## For inference

If you're just interested in producing speech representations from audio using
a pre-trained model, install via

``` sh
pip install git+https://github.com/sdrobert/scpc.git
```

Then, given a directory `in_dir` containing a bunch of `.wav` files, `.pt`
files containing speech representations will be stored in `out_dir` with
the command

``` sh
scpc-a2r <feat-args> in_dir <ckpt> out_dir
```

`<ckpt>` is the path to the pre-trained model, usually named `best.ckpt`.
For models trained on raw speech, `<feat-args>` is just `--raw`. Otherwise it's
necessary to pass at least a JSON file for configuring the feature frontend.
The relevant arguments should be stored in a file called `feats.args.txt` in
the same folder as `<ckpt>`.

Information about the pre-trained model, including the number of parameters,
downsampling factor, and so on, may be determined via the command

``` sh
scpc-info <feat-args> <ckpt> [<out-file>]
```

## For training

Training involves a few more dependencies, which can be installed via the
command

``` sh
pip install git+https://github.com/sdrobert/scpc.git[train]
```

If you want to follow the recipe in [run.sh](./run.sh), clone this repo, then

``` sh
git submodule update --init  # populates prep and s3prl
conda env create -f environment.yaml
conda activate scpc
./run.sh
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
pip install git+https://github.com/sdrobert/s3prl@scpc
```

## License

This repository is Apache 2.0-licensed. See [LICENSE](LICENSE) for the full
license details.

[s3prl](https://github.com/s3prl/s3prl) is also Apache 2.0-licensed. Our fork's
modifications are restricted solely to the [scpc upstream
folder](s3prl/s3prl/upstream/scpc).

[cpc_audio](https://github.com/facebookresearch/CPC_audio), which this
repository borrows from in [modules.py](src/scpc/modules.py), is MIT-licensed.
See [LICENSE_cpc_audio](LICENSE_cpc_audio).
