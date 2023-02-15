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

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pydrobert.torch.lightning import LitSpectDataModule

from .models import PretrainedFrontend


def main(args: Optional[Sequence[str]] = None):
    """Run scpc subcommands"""
    parser = argparse.ArgumentParser(
        description=main.__doc__, fromfile_prefix_chars="@"
    )
    subparsers = parser.add_subparsers(
        help="which routine to run", dest="cmd", required=True
    )

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="What version/run of this model to perform/continue",
    )
    LitSpectDataModule.add_argparse_args(
        fit_parser,
        read_format_str="--read-data-{file_format}",
        print_format_str="--print-data-{file_format}",
    )
    PretrainedFrontend.add_argparse_args(
        fit_parser,
        read_format_str="--read-model-{file_format}",
        print_format_str="--print-model-{file_format}",
    )
    pl.Trainer.add_argparse_args(fit_parser)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("ckpt", help="Checkpoint to load")
    predict_parser.add_argument("dir", help="Where to store predictions")
    predict_parser.add_argument(
        "--device", default=None, type=torch.device, help="Which device to run on"
    )
    predict_parser.add_argument(
        "--numpy",
        action="store_true",
        default=False,
        help="Whether to store as .npy files",
    )

    options = parser.parse_args(args)

    if options.cmd == "fit":
        data = LitSpectDataModule.from_argparse_args(
            options,
            suppress_alis=False,
            suppress_uttids=False,
            batch_first=True,
            # shuffle=False,
            pin_memory=False,
        )
        model = PretrainedFrontend.from_argparse_args(options)
        root_dir = options.default_root_dir
        if root_dir is None:
            root_dir = os.getcwd()
        model_name = model.params.name
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
        logger_dir = os.path.join(root_dir, "tb_logs")
        os.makedirs(logger_dir, exist_ok=True)
        logger = TensorBoardLogger(logger_dir, model_name, options.version)
        trainer: pl.Trainer = pl.Trainer.from_argparse_args(
            options, replace_sampler_ddp=False, callbacks=[cc], logger=logger
        )
        trainer.fit(model, data, ckpt_path="last")
        if not trainer.interrupted:
            # require training to have finished
            trainer.save_checkpoint(os.path.join(model_dir, "best.ckpt"))
    elif options.cmd == "predict":
        raise NotImplementedError
        # XXX(sdrobert): we handle this loop ourselves so no ddp stuff is going on.
        device = options.device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PretrainedFrontend.load_from_checkpoint(options.ckpt).to(device)
        model.eval()
        data.prepare_data()
        data.setup()
        dl = data.predict_dataloader()
        os.makedirs(options.dir, exist_ok=True)
        for feats, _, _, feat_lens, _, uttids in dl:
            feats, feat_lens = feats.to(device), feat_lens.to(device)
            x, lens = model(feats, feat_lens)
            for x_n, lens_n, uttids_n in zip(x, lens, uttids):
                prefix = os.path.join(options.dir, uttids_n)
                x_n = x_n[:lens_n].cpu()
                if options.numpy:
                    np.save(prefix + ".npy", x_n.numpy())
                else:
                    torch.save(x_n, prefix + ".pt")

    else:
        raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
