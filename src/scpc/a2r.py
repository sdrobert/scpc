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

import sys
import os
import argparse
import glob

from typing import Optional, Sequence

import torch
import numpy as np

from .expert import get_feature_extractor_parser, UpstreamExpert

try:
    import pydrobert.speech.util as sutil
    import pydrobert.speech.config as sconfig
except ImportError:
    sutil = sconfig = None


def _is_dir(val: str) -> str:
    if not os.path.isdir(val):
        raise ValueError(f"'{val}' is not a directory")
    return val


@torch.no_grad()
def main(args: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convert audio files to representation files",
        parents=[get_feature_extractor_parser()],
    )
    parser.add_argument("in_dir", type=_is_dir, help="Path containing wav files")
    parser.add_argument(
        "ckpt", type=argparse.FileType("rb"), help="Path to pre-trained model"
    )
    parser.add_argument("out_dir", help="Path to store speech reps to")
    parser.add_argument(
        "--numpy",
        action="store_true",
        default=False,
        help="Whether to store as .npy files",
    )
    parser.add_argument("--audio-suffix", default=".wav", help="Suffix of audio files")
    parser.add_argument(
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
    parser.add_argument(
        "--channel",
        type=int,
        default=-1,
        help="Channel to draw audio from. Default is to assume mono",
    )

    options = parser.parse_args(args)

    expert = UpstreamExpert(options.ckpt, options=options)

    for inf in glob.iglob(f"{glob.escape(options.in_dir)}/**/*{options.audio_suffix}"):
        utt_id = os.path.basename(inf).rsplit(options.audio_suffix, 1)[0]
        signal = sutil.read_signal(inf, dtype=np.float32, force_as=options.force_as)
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
