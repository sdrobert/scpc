# scpc

This is the official repository of the paper Bigger is not Always Better: The Effect of Context Size on Speech Pre-Training. The repo contains the source
for all experiments from the paper, training, and inference. All artifacts,
including model checkpoints, are stored elsewhere, to be included upon submission.

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

As mentioned, the pre-trained models we made are stored in a HuggingFace repo.
They may be downloaded individually via the `huggingface_hub` package 

Information about the pre-trained model, including the number of parameters,
downsampling factor, and so on, may be determined via the command

``` sh
scpc-info <feat-args> <ckpt> [<out-file>]
```

## For pre-training

Pre-training involves a few more dependencies, which can be installed via the
command

``` sh
pip install git+https://github.com/sdrobert/scpc.git[train]
```

This gives you access to the command

``` sh
scpc-train [--read-model-yaml <model-yaml>] <train-dir> <val-dir>
```

Where `<model-yaml>` is a path to a model YAML file which configures the
training session, such as any of the files `conf/model.*.yaml`, and
`{train,val}-dir` are the `pydrobert.torch.data.SpectDataSet` directories of
the training and validation sets. The SpectDataSet directories should have
already been passed through the feature frontend; adapt `<feat-args>` to the
command
[signals-to-torch-feat-data-dir](https://pydrobert-speech.readthedocs.io/en/latest/cli.html#signals-to-torch-feat-dir).
Since these are pre-training objectives, directories don't need reference
transcriptions nor alignments[^1]. By default, training artifacts and the final
checkpoint are stored in `./<model-name>/`. The resulting file `best.ckpt`
contains its relevant configuration and can thus be separated from other files
during inference.

## For the Zero Resource Speech Challenge, task 1

[ZeroSpeech](https://zerospeech.com/) doesn't need anything more from this
package than `scpc-a2r`. Merely follow the
[how-to](https://zerospeech.com/tasks/task_1/how_to/) and then run `scpc-a2r`
on the directories containing the wav files, using the flag `--numpy` to save
representations as Numpy arrays, and set the destination to the submission
directory. Then continue with scoring.

## Running the recipes

Rather than run a series of the bare commands above, our experimentation, much
like [kaldi](https://kaldi-asr.org/), organizes experiments _viz_ BASH recipe
scripts. There are recipes for training ([run.sh](./run.sh)), ZeroSpeech
ABX-LS ([zrc_run.sh](./scripts/zrc_run.sh)), SUPERB
([superb_run.sh](./scripts/superb_run.sh)), and our own ASR benchmark
[baseline_run.sh](./scripts/baseline_run.sh).

To get started with the recipes, clone this repo, then run the following
(assuming [conda](https://conda.org/) has already been installed):

``` sh
git submodule update --init  # populates prep and s3prl
conda env create -f environment.yaml  # creates the conda env 'scpc'
conda activate scpc  # activates that environment
```

The `scpc` environment comes installed with all packages we used in
development. Not all of them will be necessary for all recipes. For example
`zrc-benchmark` is not needed outside `zrc_run.sh`.

In general, recipes spawn two directories of note. `data/` contains the
relevant audio files, features, transcriptions, _etc._ while `exp/` contains
models, scores, _etc._. The structure of `exp/` after training is as follows:

```text
exp/
  <model-name>/
    version_<version-no>/
      ...
      expert.args.full.txt
      best.ckpt
      model.yaml
  ...
```

`<version-no>` is used to keep track of the model version. If we updated the
learning rate, for example, we would update the model configuration and bump
the version. To keep track of which configuration was used for each model
version, `conf/model.<model-name>.yaml` is saved to
`exp/<model-name>/version_<version-no>/model.yaml` on the first reference to
the specific version number. Thereonafter `model.yaml` is used to configure
the version, allowing the file in `conf/` to be changed. `expert.args.full.txt`
serves as the file `<expert-config>`. However, unlike models/training, features
have no notion of versioning: updating the feature frontend will result in a
mismatch between past and future training/inference. If a new configuration is
desired, it's best to name that configuration something else.

The HuggingFace repo stores the contents of the `exp/` folder over the course
of our experimentation. You may download all such files by cloning the repo:

``` sh
redacted for anonymity
```

To learn how to configure different models, training partitions, and so on,
run

``` sh
./run.sh -h  # or some other run script
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

[^1]: Unless the chunking policy specified by the configuration is one of
      `'ali'` or `'ref'`.
