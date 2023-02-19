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

from .models import PretrainedFrontend


def main(args: Optional[Sequence[str]] = None):
    """Run scpc subcommands"""
    parser = argparse.ArgumentParser(
        description=main.__doc__, fromfile_prefix_chars="@"
    )
    subparsers = parser.add_subparsers(
        help="which routine to run", dest="cmd", required=True
    )

    PretrainedFrontend.add_argparse_args(
        parser,
        read_format_str="--read-model-{file_format}",
        print_format_str="--print-model-{file_format}",
    )

    fit_parser = subparsers.add_parser("fit", help="Train a model")
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
    pl.Trainer.add_argparse_args(fit_parser)
    fit_parser.add_argument(
        "--num-workers", type=int, default=None, help="Number of workers in datasets"
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Output hidden states of trained model"
    )
    predict_parser.add_argument(
        "pt", type=argparse.FileType("rb"), help="Path to state_dict to load"
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

    info_parser = subparsers.add_parser(
        "info", help="Print info about model based on configuration"
    )
    info_parser.add_argument(
        "out",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Where to write info to. Default is stdout",
    )

    options = parser.parse_args(args)

    plm = PretrainedFrontend.from_argparse_args(options)
    if options.cmd == "fit":
        data = LitSpectDataModule.from_argparse_args(
            options,
            suppress_alis=False,
            suppress_uttids=False,
            batch_first=True,
            # shuffle=False,
            pin_memory=False,
            num_workers=options.num_workers,
        )
        root_dir = options.default_root_dir
        if root_dir is None:
            root_dir = os.getcwd()
        model_name = plm.params.name
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
        trainer.fit(plm, data, ckpt_path="last")
        if not trainer.interrupted:
            # require training to have finished
            plm = PretrainedFrontend.load_from_checkpoint(
                cc.best_model_path, params=plm.params
            )
            torch.save(plm.get_model().state_dict(), os.path.join(model_dir, "best.pt"))
    elif options.cmd == "predict":
        # XXX(sdrobert): we handle this loop ourselves so no ddp stuff is going on.
        device = options.device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        options.params.initialize_missing()
        model = plm.get_model().to(device)
        del plm
        state_dict = torch.load(options.pt, device)
        model.load_state_dict(state_dict)
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
        out = options.out
        out.write(f"input_size {plm.input_size}\n")
        out.write(f"output_size {plm.output_size}\n")
        out.write(f"downsampling_factor {plm.downsampling_factor}\n")
        out.write(f"num_model_params {plm.num_model_parameters}\n")
        out.write(f"num_total_params {plm.num_total_parameters}\n")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
