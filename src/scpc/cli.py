# Copyright 2023 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
from typing import Optional, Sequence

import torch
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pydrobert.torch.lightning import LitSpectDataModule
from pydrobert.torch.data import SpectDataSet

from .models import LightningPretrainedFrontend
from .modules import Encoder


def main(args: Optional[Sequence[str]] = None):
    """Run scpc subcommands"""
    parser = argparse.ArgumentParser(
        description=main.__doc__, fromfile_prefix_chars="@"
    )
    subparsers = parser.add_subparsers(
        help="which routine to run", dest="cmd", required=True
    )

    fit_parser = subparsers.add_parser("fit", help="Train a model")
    fit_parser.add_argument("train_dir")
    fit_parser.add_argument("val_dir")
    fit_parser.add_argument(
        "best",
        nargs="?",
        default=None,
        help="Where to save best inference model chekpoint to. Defaults to "
        "'<root_dir>/<model_name>/version_<version>/best.ckpt' if --version is "
        "specified, otherwise '<root_dir>/<model_name>/best.ckpt'",
    )
    fit_parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="What version/run of this model to perform/continue",
    )
    LightningPretrainedFrontend.add_argparse_args(
        fit_parser,
        read_format_str="--read-model-{file_format}",
        print_format_str="--print-model-{file_format}",
    )
    pl.Trainer.add_argparse_args(fit_parser)
    fit_parser.add_argument(
        "--num-workers", type=int, default=None, help="Number of workers in datasets"
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Output hidden states of trained model"
    )
    predict_parser.add_argument(
        "best", type=argparse.FileType("rb"), help="Path to best checkpoint"
    )
    predict_parser.add_argument("in_dir", help="Path to SpectDataSet dir to read from")
    predict_parser.add_argument("out_dir", help="Where to store representations")
    predict_parser.add_argument(
        "--device", default=None, type=torch.device, help="Which device to run on"
    )
    predict_parser.add_argument(
        "--numpy",
        action="store_true",
        default=False,
        help="Whether to store as .npy files",
    )

    info_parser = subparsers.add_parser("info", help="Print info about inference model")
    info_parser.add_argument(
        "best", type=argparse.FileType("rb"), help="Path to checkpoint"
    )
    info_parser.add_argument(
        "out",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Where to write info to. Default is stdout",
    )

    options = parser.parse_args(args)

    if options.cmd == "fit":
        lpf = LightningPretrainedFrontend.from_argparse_args(options)
        dparams = lpf.params.training.data
        assert dparams is not None
        if dparams.train_dir is not None:
            logging.getLogger("pytorch_lightning").warn(
                "params.training.data.train_dir has been specified. Overwriting with "
                "command line argument"
            )
        if dparams.val_dir is not None:
            logging.getLogger("pytorch_lightning").warn(
                "params.training.data.val_dir has been specified. Overwriting with "
                "command line argument"
            )
        dparams.train_dir = options.train_dir
        dparams.val_dir = options.val_dir
        data = LitSpectDataModule(
            dparams,
            batch_first=True,
            sort_batch=False,
            suppress_alis=False,
            tokens_only=False,
            suppress_uttids=False,
            shuffle=None,
            num_workers=options.num_workers,
            warn_on_missing=False,
            on_uneven_distributed="raise",
        )
        root_dir = options.default_root_dir
        if root_dir is None:
            root_dir = os.getcwd()
        model_name = lpf.params.name
        if model_name == "model_params":
            logging.getLogger("pytorch_lightning").warn(
                "saving with default model name 'model_name'. Use a non-default "
                "name to avoid clobbering with different configurations"
            )
        model_dir = os.path.join(root_dir, model_name)
        if options.version is not None:
            model_dir = os.path.join(model_dir, f"version_{options.version}")
        os.makedirs(model_dir, exist_ok=True)
        cc = ModelCheckpoint(model_dir, save_last=True)
        callbacks = [cc]
        if options.enable_progress_bar:
            callbacks.append(RichProgressBar())
        logger_dir = os.path.join(root_dir, "tb_logs")
        os.makedirs(logger_dir, exist_ok=True)
        logger = TensorBoardLogger(logger_dir, model_name, options.version)
        trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            options, replace_sampler_ddp=False, callbacks=callbacks, logger=logger
        )
        trainer.fit(lpf, data, ckpt_path="last")
        # require training to have finished before saving
        if not trainer.interrupted:
            lpf = LightningPretrainedFrontend.load_from_checkpoint(
                cc.best_model_path, params=lpf.params
            )
            ckpt_path = options.best
            if ckpt_path is None:
                ckpt_path = os.path.join(model_dir, "best.ckpt")
            lpf.get_inference_model().save_checkpoint(ckpt_path)
    elif options.cmd == "predict":
        # XXX(sdrobert): we handle this loop ourselves so no ddp stuff is going on.
        device = options.device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Encoder.from_checkpoint(options.best).to(device)
        model.eval()
        ds = SpectDataSet(options.in_dir, suppress_alis=True, suppress_uttids=False)
        os.makedirs(options.out_dir, exist_ok=True)
        with torch.no_grad():
            for feats, _, utt_id in ds:
                prefix = os.path.join(options.out_dir, utt_id)
                feats = feats.to(device).unsqueeze(0)
                x, lens = model(feats)
                assert lens is None
                x = x.squeeze(0).cpu()
                if options.numpy:
                    np.save(prefix + ".npy", x.double().numpy())
                else:
                    torch.save(x, prefix + ".pt")
    elif options.cmd == "info":
        encoder = Encoder.from_checkpoint(options.best)
        out = options.out
        out.write(f"input_size {encoder.input_size}\n")
        out.write(f"output_size {encoder.output_size}\n")
        out.write(f"downsampling_factor {encoder.downsampling_factor}\n")
        out.write(f"num_params {sum(np.prod(p.shape) for p in encoder.parameters())}\n")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
