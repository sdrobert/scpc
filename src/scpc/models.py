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
import itertools
import logging
import re

from typing import Dict, Iterator, Literal, Optional, Tuple

import torch
import param
import numpy as np
import pytorch_lightning as pl
import pydrobert.param.argparse as pargparse

from pydrobert.torch.lightning import LitSpectDataModuleParams
from pydrobert.torch.modules import ChunkBySlices, SliceSpectData

from .modules import *

__all__ = [
    "CausalSelfAttentionEncoderParams",
    "ChunkingParams",
    "ConvEncoderParams",
    "CPCLossParams",
    "FeedForwardEncoderParams",
    "LightningPretrainedFrontend",
    "LightningPretrainedFrontendParams",
    "RecurrentEncoderParams",
    "SelfAttentionEncoderParams",
    "TrainingParams",
]


class ConvEncoderParams(param.Parameterized):
    output_size: int = param.Integer(
        512, bounds=(1, None), doc="Size of output (and # of conv output channels)"
    )
    norm_type: Literal["none", "batch" "channel", "instance"] = param.ObjectSelector(
        "none",
        ["none", "batch", "channel", "instance"],
        doc="Intermediate layer norm to apply. 'none' is none. 'batch' normalizes over"
        "batch elements; 'channel' normalizes over channels; 'instance' normalizes "
        "over samples",
    )


class FeedForwardEncoderParams(param.Parameterized):
    output_size: int = param.Integer(256, bounds=(1, None), doc="Size of the output")
    nonlin_type: Literal["relu", "sigmoid", "tanh", "none"] = param.ObjectSelector(
        "relu",
        ["relu", "sigmoid", "tanh", "none"],
        doc="Type of nonlinearity to apply after linear transform",
    )
    bias: bool = param.Boolean(True, doc="Whether to use a bias vector")


class SelfAttentionEncoderParams(param.Parameterized):
    num_layers: int = param.Integer(
        1, bounds=(1, None), doc="Number of self-attention layers"
    )
    num_heads: int = param.Integer(1, bounds=(1, None), doc="Number of attention heads")
    dim_feedforward: int = param.Integer(
        1, bounds=(1, None), doc="Size of intermediate representation"
    )


class CausalSelfAttentionEncoderParams(SelfAttentionEncoderParams):
    max_width: Optional[int] = param.Integer(
        None,
        bounds=(1, None),
        doc="Max number of past frames to attend to. Unspecified = all",
    )


class RecurrentEncoderParams(param.Parameterized):
    output_size: int = param.Integer(
        256, bounds=(1, None), doc="Size of output (and hidden size)"
    )
    num_layers: int = param.Integer(
        1, bounds=(1, None), doc="Number of recurrent layers"
    )
    recurrent_type: Literal["gru", "lstm", "rnn"] = param.ObjectSelector(
        "gru", ["gru", "lstm", "rnn"], doc="Type of recurrent cell"
    )


class CPCLossParams(param.Parameterized):
    prediction_steps: int = param.Integer(
        12, bounds=(1, None), doc="Number of frames ahead to try and predict"
    )
    negative_samples: int = param.Integer(
        128, bounds=(1, None), doc="Number of negative samples to estimate with"
    )
    num_speakers: Optional[int] = param.Integer(
        None,
        bounds=(1, None),
        doc="Number of speakers to construct embeddings for. Source hashes will be "
        "extracted from utterance ids. Unset means no embeddings used",
    )
    speaker_regex: str = param.String(
        r"^lbi-([^-]+)-.*$",
        doc="Regular expression to extract speaker id from utterance id. Must contain "
        "one group to be used as the speaker id",
    )
    prediction_type: Literal["ff", "recur", "csa"] = param.ObjectSelector(
        "ff",
        ["ff", "recur", "csa"],
        doc="Type of prediction network to use. 'ff' is "
        "a matrix (original); 'recur' is a single-layer LSTM; 'csa' is a causal "
        "transformer",
    )


class ChunkingParams(param.Parameterized):
    policy: Literal["fixed", "none", "ali", "ref"] = param.ObjectSelector(
        "fixed",
        ["fixed", "none", "ali", "ref"],
        doc="Chunking policy. 'none' for no chunking; 'fixed' for fixed window; 'ali' "
        "for sequential segments in utterance partition (requires ali/); and "
        "'ref' for possibly overlapping segments in utterance (requires ref/)",
    )
    window_type: Literal["causal", "symmetric", "future"] = param.ObjectSelector(
        "causal",
        ["causal", "symmetric", "future"],
        doc="Chunking window type. 'causal' uses past frames; 'future' uses future "
        "frames; 'symmetric' uses both (doubling window size)",
    )
    lobe_size: int = param.Integer(
        20479,
        bounds=(0, None),
        doc="Size of chunking window lobe. Frames for 'fixed' and 'ref' policies; "
        "segments for 'ali'",
    )
    pad_mode: Literal[
        "valid", "constant", "reflect", "replicate"
    ] = param.ObjectSelector(
        "valid",
        ["valid", "constant", "reflect", "replicate"],
        doc="How to pad chunks exceeding utterance boundaries. 'valid' means throw "
        "away any such chunks; 'constant' pad with pad_constant; 'reflect' pad with "
        "values reflected around boundaries; 'replicate' pad by replicating boundary "
        "values",
    )
    pad_constant: float = param.Number(
        0.0, doc="Value to pad with when pad_mode is 'constant'"
    )

    max_chunks: Optional[int] = param.Integer(
        8,
        allow_None=True,
        bounds=(1, None),
        doc="Maximum number of chunks per batch. If the total number of extracted "
        "chunks exceeds this number, a random assortment of them is kept. If unset, "
        "no maximum is set",
    )


class TrainingParams(param.Parameterized):
    data: Optional[LitSpectDataModuleParams] = param.ClassSelector(
        LitSpectDataModuleParams, instantiate=False, doc="Data parameters"
    )

    optimizer: Literal["adam", "sgd"] = param.ObjectSelector(
        "adam", ["adam", "sgd"], doc="Optimizer class"
    )
    learning_rate: float = param.Number(
        2e-4, bounds=(0, None), inclusive_bounds=(False, True), doc="Learning rate"
    )
    dropout_prob: float = param.Magnitude(0.0, doc="Probability of dropping out a unit")

    chunking: Optional[ChunkingParams] = param.ClassSelector(
        ChunkingParams, instantiate=False, doc="Parameters for chunking"
    )

    loss_type: Literal["cpc"] = param.ObjectSelector(
        "cpc", ["cpc"], doc="Loss to train model with"
    )
    cpc_loss: Optional[CPCLossParams] = param.ClassSelector(
        CPCLossParams,
        instantiate=False,
        doc="Parameters for CPC loss (if loss_type = 'cpc')",
    )

    batch_size: int = param.Integer(1, bounds=(1, None), doc="Size of")

    def initialize_missing(self):
        if self.data is None:
            self.data = LitSpectDataModuleParams(name="data")
            self.data.prefer_split = False
            self.data.initialize_missing()
            self.data.train_params.batch_size = 1
            self.data.train_params.drop_last = True
        if self.chunking is None:
            self.chunking = ChunkingParams(name="chunking")
        if self.cpc_loss is None:
            self.cpc_loss = CPCLossParams(name="cpc_loss")


class LightningPretrainedFrontendParams(param.Parameterized):
    system_description: str = param.String(
        "", doc="Description of the system for ZRC submission"
    )

    input_size: int = param.Integer(
        1, bounds=(1, None), doc="Size of input feature dimension (1 for raw)"
    )

    latent_type: Literal["conv", "ff", "id"] = param.ObjectSelector(
        "conv",
        ["conv", "ff", "id"],
        doc="Which encoder to use for the 'latent' part of the model. 'conv' is "
        "convolutional; 'ff' is feed-forward; 'id' is identity (noop)",
    )
    context_type: Literal["csa", "sa", "recur", "id"] = param.ObjectSelector(
        "recur",
        ["csa", "sa", "recur", "id"],
        doc="Which encoder to use for the 'context' part of the model. 'csa' is "
        "causal self-atttention; 'sa' is self-attention (non-causal); 'recur' is "
        "recurrent; 'id' is identity (noop)",
    )

    conv: Optional[ConvEncoderParams] = param.ClassSelector(
        ConvEncoderParams,
        instantiate=False,
        doc="Parameters for latent convolutional encoder (if latent_type = 'conv')",
    )
    ff: Optional[FeedForwardEncoderParams] = param.ClassSelector(
        FeedForwardEncoderParams,
        instantiate=False,
        doc="Parameters for latent feed-forward encoder (if latent_type = 'ff')",
    )
    csa: Optional[CausalSelfAttentionEncoderParams] = param.ClassSelector(
        CausalSelfAttentionEncoderParams,
        instantiate=False,
        doc="Parameters for context causal self-attention encoder "
        "(if context_type = 'csa')",
    )
    sa: Optional[SelfAttentionEncoderParams] = param.ClassSelector(
        SelfAttentionEncoderParams,
        instantiate=False,
        doc="Parameters for context self-attention encoder (if context_type = 'sa')",
    )
    recur: Optional[RecurrentEncoderParams] = param.ClassSelector(
        RecurrentEncoderParams,
        instantiate=False,
        doc="Parameters for context recurrent encoder (if context_type = 'recur')",
    )

    training: Optional[TrainingParams] = param.ClassSelector(
        TrainingParams, instantiate=False, doc="Parameters for training"
    )

    def initialize_missing(self):
        if self.conv is None:
            self.conv = ConvEncoderParams(name="conv")
        if self.ff is None:
            self.ff = FeedForwardEncoderParams(name="ff")
        if self.csa is None:
            self.csa = CausalSelfAttentionEncoderParams(name="csa")
        if self.sa is None:
            self.sa = SelfAttentionEncoderParams(name="sa")
        if self.recur is None:
            self.recur = RecurrentEncoderParams(name="recur")
        if self.training is None:
            self.training = TrainingParams(name="training")
        self.training.initialize_missing()


Batch = Tuple[
    torch.Tensor,  # feats
    Optional[torch.Tensor],  # alis
    Optional[torch.Tensor],  # refs
    Optional[torch.Tensor],  # feat_sizes
    Optional[torch.Tensor],  # ref_sizes
    Tuple[str, ...],  # utt_ids
]


class LightningPretrainedFrontend(pl.LightningModule):
    params: LightningPretrainedFrontendParams
    latent: Encoder
    context: Encoder
    slicer: Optional[SliceSpectData]
    chunker: Optional[ChunkBySlices]

    _speaker_regex: Optional[re.Pattern[str]]
    _speaker_map: Dict[str, int]
    _speaker_min: int

    def __init__(self, params: LightningPretrainedFrontendParams) -> None:
        super().__init__()
        self.params = params
        self._speaker_map = dict()
        self._speaker_min = -1

        if params.training is None:
            raise ValueError("params.training not initialized")
        if params.training.chunking is None:
            raise ValueError("params.training.chunking not initialized")

        if params.latent_type == "conv":
            if params.conv is None:
                raise ValueError("params.conv not initialized")
            self.latent = ConvEncoder(
                params.input_size,
                params.conv.output_size,
                params.conv.norm_type,
                params.training.dropout_prob,
            )
        elif params.latent_type == "ff":
            if params.ff is None:
                raise ValueError("params.ff not initialized")
            self.latent = FeedForwardEncoder(
                params.input_size,
                params.ff.output_size,
                params.ff.nonlin_type,
                params.ff.bias,
                params.training.dropout_prob,
            )
        elif params.latent_type == "id":
            self.latent = IdentityEncoder(params.input_size)
        else:
            raise NotImplementedError

        if params.context_type == "csa":
            if params.csa is None:
                raise ValueError("params.csa not initialized")
            self.context = CausalSelfAttentionEncoder(
                self.latent.output_size,
                None,
                params.csa.max_width,
                params.csa.num_layers,
                params.csa.num_heads,
                params.csa.dim_feedforward,
                params.training.dropout_prob,
                params.training.chunking.policy != "fixed",
            )
        elif params.context_type == "sa":
            if params.sa is None:
                raise ValueError("params.sa not initialized")
            self.context = SelfAttentionEncoder(
                self.latent.output_size,
                params.sa.num_layers,
                params.sa.num_heads,
                params.sa.dim_feedforward,
                params.training.dropout_prob,
                params.training.chunking.policy != "fixed",
            )
        elif params.context_type == "recur":
            if params.recur is None:
                raise ValueError("params.recur not initialized")
            self.context = RecurrentEncoder(
                self.latent.output_size,
                params.recur.output_size,
                params.recur.num_layers,
                params.recur.recurrent_type,
                params.training.dropout_prob,
            )
        elif params.context_type == "id":
            self.context = IdentityEncoder(self.latent.output_size)
        else:
            raise NotImplementedError

        if params.training.loss_type == "cpc":
            if params.training.cpc_loss is None:
                raise ValueError("params.training.cpc_loss not initialized")
            num_speakers = params.training.cpc_loss.num_speakers
            if num_speakers is not None:
                self._speaker_regex = re.compile(params.training.cpc_loss.speaker_regex)
                if self._speaker_regex.groups != 1:
                    raise ValueError(
                        f"expected one group in regex '{self._speaker_regex.pattern}'; "
                        f"got {self._speaker_regex.groups}"
                    )
            else:
                self._speaker_regex = None
            if params.training.cpc_loss.prediction_type == "csa":
                penc = CausalSelfAttentionEncoder(
                    self.context.output_size,
                    params.training.cpc_loss.prediction_steps * self.latent.output_size,
                    dropout_prob=params.training.dropout_prob,
                )
            elif params.training.cpc_loss.prediction_type == "recur":
                penc = RecurrentEncoder(
                    self.context.output_size,
                    params.training.cpc_loss.prediction_steps * self.latent.output_size,
                    recurrent_type="lstm",
                    dropout_prob=params.training.dropout_prob,
                )
            else:
                penc = FeedForwardEncoder(
                    self.context.output_size,
                    params.training.cpc_loss.prediction_steps * self.latent.output_size,
                    "none",
                    False,
                    params.training.dropout_prob,
                )
            self.cpc_loss = CPCLossNetwork(
                self.latent.output_size,
                self.context.output_size,
                params.training.cpc_loss.prediction_steps,
                params.training.cpc_loss.negative_samples,
                penc,
                num_speakers,
            )
        else:
            raise NotImplementedError

        if params.training.chunking.policy != "none":
            pad_mode = params.training.chunking.pad_mode
            is_valid = False
            if pad_mode == "valid":
                pad_mode = "constant"
                is_valid = True
            self.slicer = SliceSpectData(
                params.training.chunking.policy,
                params.training.chunking.window_type,
                is_valid,
                params.training.chunking.lobe_size,
            )
            self.chunker = ChunkBySlices(
                pad_mode, params.training.chunking.pad_constant
            )
        else:
            self.register_module("slicer", None)
            self.register_module("chunker", None)

    @property
    def input_size(self) -> int:
        return self.params.input_size

    @property
    def output_size(self) -> int:
        return self.context.output_size

    @property
    def training_is_fixed_width(self) -> bool:
        return self.params.training.chunking.policy == "fixed"

    @property
    def downsampling_factor(self) -> int:
        return self.context.downsampling_factor * self.latent.downsampling_factor

    @staticmethod
    def _num_params(ps: Iterator[torch.Tensor]) -> int:
        return sum(np.prod(p.shape) for p in ps)

    @property
    def num_model_parameters(self) -> int:
        return self._num_params(
            itertools.chain(self.latent.parameters(), self.context.parameters())
        )

    @property
    def num_total_parameters(self) -> int:
        return self._num_params(self.parameters())

    def get_inference_model(self) -> EncoderSequence:
        return EncoderSequence(self.latent, self.context)

    def get_speakers_from_uttids(self, utt_ids: Tuple[str, ...]) -> torch.Tensor:
        speakers = torch.empty(len(utt_ids), dtype=torch.long)
        if self._speaker_regex is None:
            return speakers
        num_speakers = self.params.training.cpc_loss.num_speakers
        assert num_speakers is not None
        for n, utt_id in enumerate(utt_ids):
            speaker = self._speaker_regex.match(utt_id).group(0)
            id_ = self._speaker_map.get(speaker, None)
            if id_ is None:
                self._speaker_min += 1
                if self._speaker_min >= num_speakers:
                    logger = logging.getLogger("pytorch_lightning")
                    logger.warn(f"number of speakers exceeded {num_speakers}")
                    self._speaker_min = num_speakers
                id_ = self._speaker_min
            speakers[n] = id_
        return speakers

    def pretrain_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        feats, _, _, feat_sizes, _, utt_ids = batch
        if len(utt_ids) == 0:
            return feats.new_zeros(1)
        speakers = self.get_speakers_from_uttids(utt_ids).to(feats.device)
        latent, lens = self.latent(feats, feat_sizes)
        context, lens = self.context(latent, lens)
        if self.params.training.loss_type == "cpc":
            loss = self.cpc_loss(latent, context, lens, speakers)
        else:
            raise NotImplementedError
        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss = self.pretrain_step(batch, batch_idx)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss = self.pretrain_step(batch, batch_idx)
        self.log("val_loss", loss, batch_size=batch[0].size(0), sync_dist=True)
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss = self.pretrain_step(batch, batch_idx)
        self.log("test_loss", loss, batch_size=batch[0].size(0), sync_dist=True)
        return loss

    def on_before_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        feats, alis, refs, feat_sizes, ref_sizes, utt_ids = batch
        policy = self.params.training.chunking.policy
        if policy != "none":
            if policy == "fixed":
                slices, sources = self.slicer(feats, feat_sizes)
            elif policy == "ali":
                if alis is None:
                    raise ValueError("chunking policy is 'ali' but alis is None")
                slices, sources = self.slicer(alis, feat_sizes)
            else:
                if refs is None or ref_sizes is None:
                    raise ValueError(
                        "chunking policy is 'ref' but refs or ref_sizes is None"
                    )
                slices, sources = self.slicer(refs, ref_sizes, feat_sizes)
            feats, feat_sizes = self.chunker(
                feats[sources], slices, feat_sizes[sources]
            )
            utt_ids = tuple(utt_ids[src] for src in sources)
            max_chunks, N = self.params.training.chunking.max_chunks, feats.size(0)
            if max_chunks is not None and max_chunks < N:
                sources = torch.randperm(N)[:max_chunks]
                utt_ids = tuple(utt_ids[src] for src in sources)
                feats, feat_sizes = feats[sources], feat_sizes[sources]
            if policy == "fixed":
                feat_sizes = None
            del sources, slices
        return feats, None, None, feat_sizes, None, utt_ids

    def forward(
        self,
        feats: torch.Tensor,
        feat_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.get_inference_model().forward(feats, feat_lens)

    def predict_step(
        self, batch: Batch, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self(batch[0], batch[3])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.params.training.optimizer == "adam":
            Optim = torch.optim.Adam
        elif self.params.training.optimizer == "sgd":
            Optim = torch.optim.SGD
        else:
            raise NotImplementedError

        return Optim(self.parameters(), lr=self.params.training.learning_rate)

    @classmethod
    def add_argparse_args(
        cls,
        parser: argparse.ArgumentParser,
        read_format_str: str = "--read-model-{file_format}",
        print_format_str: Optional[str] = None,
    ):
        params = LightningPretrainedFrontendParams(name="model_params")
        params.initialize_missing()

        if print_format_str is not None:
            pargparse.add_serialization_group_to_parser(
                parser, params, reckless=True, flag_format_str=print_format_str
            )

        grp = pargparse.add_deserialization_group_to_parser(
            parser,
            params,
            "params",
            reckless=True,
            flag_format_str=read_format_str,
        )
        return grp

    @classmethod
    def from_argparse_args(cls, namespace: argparse.Namespace, **kwargs):
        params = namespace.params
        params.initialize_missing()
        return cls(params, **kwargs)
