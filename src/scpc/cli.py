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
import os
import sys
from typing import Optional, Sequence

import torch
import numpy as np
import pytorch_lightning as pl

from pydrobert.torch.lightning import LitSpectDataModule

from .models import PretrainedFrontend


def main(args: Optional[Sequence[str]] = None):
    """Run scpc subcommands"""
    parser = argparse.ArgumentParser(
        description=main.__doc__, fromfile_prefix_chars="@"
    )
    LitSpectDataModule.add_argparse_args(
        parser,
        read_format_str="--read-data-{file_format}",
        print_format_str="--print-data-{file_format}",
    )
    subparsers = parser.add_subparsers(
        help="which routine to run", dest="cmd", required=True
    )

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("ckpt", help="Where to store best model checkpoint")
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

    data = LitSpectDataModule.from_argparse_args(
        options,
        suppress_alis=False,
        suppress_uttids=False,
        batch_first=True,
        # shuffle=False,
        pin_memory=False,
    )

    if options.cmd == "fit":
        model = PretrainedFrontend.from_argparse_args(options)
        trainer = pl.Trainer.from_argparse_args(options, replace_sampler_ddp=False)
        trainer.fit(model, data)
        trainer.save_checkpoint(options.ckpt)
    elif options.cmd == "predict":
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
