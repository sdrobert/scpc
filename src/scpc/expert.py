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
import json
import os
import sys
import glob

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np


try:
    import pydrobert.speech.torch as storch
    import pydrobert.speech.pre as spre
    import pydrobert.speech.post as spost
    import pydrobert.speech.compute as scompute
    import pydrobert.speech.alias as salias
    import pydrobert.speech.util as sutil
    import pydrobert.speech.config as sconfig
except ImportError:
    storch = spre = spost = scompute = salias = sutil = sconfig = None

from .modules import Encoder


def get_feature_extractor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@", add_help=False)
    parser.add_argument(
        "--preprocess",
        default="[]",
        help="JSON list of configurations for "
        "``pydrobert.speech.pre.PreProcessor`` objects. Audio will be "
        "preprocessed in the same order as the list",
    )
    parser.add_argument(
        "--postprocess",
        default="[]",
        help="JSON List of configurations for "
        "``pydrobert.speech.post.PostProcessor`` objects. Features will be "
        "postprocessed in the same order as the list",
    )
    feat_group = parser.add_mutually_exclusive_group(required=True)
    feat_group.add_argument(
        "--computer-json",
        default=None,
        help="Path to JSON configuration of a feature computer",
    )
    feat_group.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="If specified, audio will not be processed",
    )
    return parser


class RawComputer(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)


def get_feature_extractor_from_opts(options) -> Tuple[torch.nn.Module, int]:
    if options.raw:
        computer, rate = RawComputer(), 1
    else:
        with open(options.computer_json) as f:
            json_ = json.load(f)
        computer = salias.alias_factory_subclass_from_arg(scompute.FrameComputer, json_)
        rate = computer.frame_shift
        if isinstance(computer, scompute.SIFrameComputer):
            computer = storch.PyTorchSIFrameComputer.from_si_frame_computer(computer)
        elif isinstance(computer, scompute.STFTFrameComputer):
            computer = storch.PyTorchSTFTFrameComputer.from_stft_frame_computer(
                computer
            )
        else:
            raise NotImplementedError(
                f"Unknown computer type {type(computer).__name__}"
            )
    computers = [computer]

    json_ = json.loads(options.preprocess)
    if len(json_):
        for json_i in json_[::-1]:
            preprocessor = salias.alias_factory_subclass_from_arg(
                spre.PreProcessor, json_i
            )
            if isinstance(preprocessor, spre.Dither):
                preprocessor = storch.PyTorchDither.from_dither(preprocessor)
            elif isinstance(preprocessor, spre.Preemphasize):
                preprocessor = storch.PyTorchPreemphasize.from_preemphasize(
                    preprocessor
                )
            else:
                raise NotImplementedError(
                    f"Unknown preprocessor type {type(preprocessor).__name__}"
                )
            computers.insert(0, preprocessor)

    json_ = json.loads(options.postprocess)
    if len(json_):
        computers.extend(
            storch.PyTorchPostProcessorWrapper.from_postprocessor(
                salias.alias_factory_subclass_from_arg(spost.PostProcessor, x)
            )
            for x in json_
        )

    if len(computers) > 1:
        computer = torch.nn.Sequential(computers)

    return computer, rate


class UpstreamExpert(torch.nn.Module):
    encoder: Encoder
    feat_extractor: torch.nn.Module

    def __init__(
        self, ckpt: str, model_config: Optional[str] = None, options=None, **kwargs
    ):
        super().__init__()
        if model_config is not None:
            if options is not None:
                raise ValueError("Cannot specify both model_config and options")
            options = get_feature_extractor_parser().parse_args(["@" + model_config])
        self.feat_extractor, self._feds_rate = get_feature_extractor_from_opts(options)
        self.name = "[scpc]"
        self.encoder = Encoder.from_checkpoint(ckpt, "cpu")

    def get_downsample_rates(self, key: str) -> int:
        rate = self.encoder.downsampling_factor
        if self.feat_extractor is not None:
            rate *= self._feds_rate
        return rate

    def forward(self, wavs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(wavs) == 0:
            return {"hidden_states": torch.empty(0, 0, self.encoder.output_size)}
        feats = [self.feat_extractor(x) for x in wavs]
        lens = torch.tensor([f.size(0) for f in feats]).to(wavs[0].device)
        x = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        x, lens = self.encoder(x, lens)
        return {"hidden_states": x}


def _is_dir(val: str) -> str:
    if not os.path.isdir(val):
        raise ValueError(f"'{val}' is not a directory")
    return val


def main(args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convert audio files to representation files",
        parents=[get_feature_extractor_parser()],
    )
    parser.add_argument("ckpt", type=argparse.FileType("rb"))
    subparsers = parser.add_subparsers(required=True, dest="cmd")

    a2r_parser = subparsers.add_parser(
        "a2r", description="Convert audio files to speech representations"
    )
    a2r_parser.add_argument("in_dir", type=_is_dir, help="Path containing wav files")
    a2r_parser.add_argument("out_dir", help="Path to store speech reps to")
    a2r_parser.add_argument(
        "--numpy",
        action="store_true",
        default=False,
        help="Whether to store as .npy files",
    )
    a2r_parser.add_argument(
        "--audio-suffix", default=".wav", help="Suffix of audio files"
    )
    a2r_parser.add_argument(
        "--force-as",
        default=None,
        choices={
            "table",
            "wav",
            "hdf5",
            "npy",
            "npz",
            "pt",
            "sph",
            "kaldi",
            "file",
            "soundfile",
        }
        | sconfig.SOUNDFILE_SUPPORTED_FILE_TYPES,
        help="Force the paths in 'map' to be interpreted as a specific type "
        "of data. table: kaldi table (key is utterance id); wav: wave file; "
        "hdf5: HDF5 archive (key is utterance id); npy: Numpy binary; npz: "
        "numpy archive (key is utterance id); pt: PyTorch binary; sph: NIST "
        "SPHERE file; kaldi: kaldi object; file: numpy.fromfile binary. soundfile: "
        "force soundfile processing.",
    )
    a2r_parser.add_argument(
        "--channel",
        type=int,
        default=-1,
        help="Channel to draw audio from. Default is to assume mono",
    )

    info_parser = subparsers.add_parser("info")
    info_parser.add_argument(
        "out", nargs="?", type=argparse.FileType("w"), default=sys.stdout
    )

    options = parser.parse_args(args)

    expert = UpstreamExpert(options.ckpt, options=options)

    if options.cmd == "info":
        out = options.out
        out.write(f"feat_size {expert.encoder.input_size}\n")
        out.write(f"output_size {expert.encoder.output_size}\n")
        out.write(f"downsampling_factor {expert.get_downsample_rates('')}\n")
        out.write(
            f"num_params {sum(np.prod(p.shape) for p in expert.encoder.parameters())}\n"
        )
    elif options.cmd == "a2r":
        with torch.no_grad():
            for inf in glob.iglob(
                f"{glob.escape(options.in_dir)}/**/*{options.audio_suffix}"
            ):
                utt_id = os.path.basename(inf).rsplit(options.audio_suffix, 1)[0]
                signal = sutil.read_signal(
                    inf, dtype=np.float32, force_as=options.force_as
                )
                if options.channel == -1 and signal.ndim > 1 and signal.shape[0] > 1:
                    raise ValueError(
                        "Utterance {}: Channel is not specified but signal has "
                        "shape {}".format(utt_id, signal.shape)
                    )
                elif (options.channel != -1 and signal.ndim == 1) or (
                    options.channel >= signal.shape[0]
                ):
                    raise ValueError(
                        "Utterance {}: Channel specified as {} but signal has "
                        "shape {}".format(utt_id, options.channel, signal.shape)
                    )
                if signal.ndim != 1:
                    signal = signal[options.channel]
                signal = torch.from_numpy(signal)
                odir = os.path.join(
                    options.out_dir,
                    os.path.relpath(os.path.dirname(inf), options.in_dir),
                )
                reps = expert([signal])["hidden_states"][0].cpu()
                os.makedirs(odir, exist_ok=True)
                if options.numpy:
                    np.save(os.path.join(odir, utt_id + ".npy"), reps.double().numpy())
                else:
                    torch.save(reps, os.path.join(odir, utt_id + ".pt"))
    else:
        raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
