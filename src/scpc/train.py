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
import sys
import os

from typing import Dict, Iterator, Literal, Optional, Tuple, Sequence

import torch
import param
import numpy as np
import pytorch_lightning as pl
import pydrobert.param.argparse as pargparse

from pydrobert.torch.lightning import LitSpectDataModuleParams, LitSpectDataModule
from pydrobert.torch.modules import ChunkBySlices, SliceSpectData
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from .modules import *


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
    num_heads: int = param.Integer(8, bounds=(1, None), doc="Number of attention heads")
    dim_feedforward: int = param.Integer(
        2048, bounds=(1, None), doc="Size of intermediate representation"
    )
    pos_period: Optional[int] = param.Integer(
        None,
        bounds=(1, None),
        doc="If set, attention mask will oscillate between [-1, 1] with this period",
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
    gutted_steps: int = param.Integer(
        0,
        bounds=(0, None),
        doc="Number of frames (< prediction_steps) to exclude from prediction",
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
    offset: int = param.Integer(
        0,
        bounds=(0, None),
        doc="Frame offset from start of utterance/chunk to begin computing loss from",
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


class BestRqLossParams(param.Parameterized):
    mask_prob: float = param.Magnitude(
        0.01, doc="Per-frame probabilty of temporal mask"
    )
    mask_width: int = param.Integer(
        40, bounds=(1, None), doc="Width (in frames) of temporal mask"
    )
    mask_mean: float = param.Number(
        None, doc="Mean of temporal mask noise. If unspecified, it's per-channel"
    )
    mask_std: float = param.Number(
        None,
        bounds=(0, None),
        doc="Stddev of temporal mask noise. If unspecified, it's per-channel",
    )
    codebook_size: int = param.Integer(
        8192, bounds=(1, None), doc="Number of quantized vectors in codebook"
    )
    codebook_dim: int = param.Integer(
        16,
        bounds=(1, None),
        doc="Size of quantized vector in codebook",
    )
    num_speakers: Optional[int] = param.Integer(
        None,
        bounds=(1, None),
        doc="Number of speakers to construct embeddings for. Source hashes will be "
        "extracted from utterance ids. Unset means no embeddings used",
    )
    offset: int = param.Integer(
        0,
        bounds=(0, None),
        doc="Frame offset from start of utterance/chunk to begin computing loss from",
    )
    speaker_regex: str = param.String(
        r"^lbi-([^-]+)-.*$",
        doc="Regular expression to extract speaker id from utterance id. Must contain "
        "one group to be used as the speaker id",
    )
    prediction_type: Literal["ff", "recur", "sa", "csa"] = param.ObjectSelector(
        "ff",
        ["ff", "recur", "sa", "csa"],
        doc="Type of prediction network to use. 'ff' is "
        "a matrix (original); 'recur' is a single-layer LSTM; 'sa' is a transformer; "
        "'csa' is a causal transformer",
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

    loss_type: Literal["cpc", "best-rq"] = param.ObjectSelector(
        "cpc", ["cpc", "best-rq"], doc="Loss to train model with"
    )
    cpc_loss: Optional[CPCLossParams] = param.ClassSelector(
        CPCLossParams,
        instantiate=False,
        doc="Parameters for CPC loss (if loss_type = 'cpc')",
    )
    best_rq_loss: Optional[BestRqLossParams] = param.ClassSelector(
        BestRqLossParams,
        instantiate=False,
        doc="Parameters for BEST-RQ loss (if loss_type = 'best-rq')",
    )

    accelerator: str = param.String("gpu", doc="Lightning accelerator")
    shuffle: Optional[bool] = param.Boolean(
        False,
        allow_None=True,
        doc="Whether to shuffle data. Unset will shuffle train but not val",
    )
    num_devices: Optional[int] = param.Integer(
        None,
        bounds=(1, None),
        doc="Number of devices to train with in distributed mode. Unset will be serial",
    )
    num_nodes: int = param.Integer(
        1,
        bounds=(1, None),
        doc="Number of nodes to train on simultaneously. Relevant only if num_devices "
        "is set",
    )
    max_epochs: int = param.Integer(
        200, bounds=(1, None), doc="The total number of epochs to train for"
    )

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
        if self.best_rq_loss is None:
            self.best_rq_loss = BestRqLossParams(name="best_rq_loss")


class LightningPretrainedFrontendParams(param.Parameterized):
    system_description: str = param.String(
        "", doc="Description of the system for ZRC submission"
    )
    feat_type: str = param.String("raw", doc="The type of feature expected as input")
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
    masker: Optional[ContiguousTemporalMask]
    cpc_loss: Optional[CPCLossNetwork]
    best_rq_loss: Optional[BestRqLossNetwork]

    _speaker_regex: Optional[re.Pattern[str]]
    _speaker_map: Dict[str, int]
    _speaker_min: int
    _num_speakers: Optional[int]

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
                params.csa.pos_period,
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
                params.csa.pos_period,
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

        self._num_speakers = self._speaker_regex = None
        if params.training.loss_type == "cpc":
            self.register_module("masker", None)
            self.register_module("best_rq_loss", None)
            if params.training.cpc_loss is None:
                raise ValueError("params.training.cpc_loss not initialized")
            self._num_speakers = params.training.cpc_loss.num_speakers
            penc_out_size = (
                params.training.cpc_loss.prediction_steps
                - params.training.cpc_loss.gutted_steps
            ) * self.latent.output_size
            if self._num_speakers is not None:
                self._speaker_regex = re.compile(params.training.cpc_loss.speaker_regex)
                if self._speaker_regex.groups != 1:
                    raise ValueError(
                        f"expected one group in regex '{self._speaker_regex.pattern}'; "
                        f"got {self._speaker_regex.groups}"
                    )
            if params.training.cpc_loss.prediction_type == "csa":
                penc = CausalSelfAttentionEncoder(
                    self.context.output_size,
                    penc_out_size,
                    dropout_prob=params.training.dropout_prob,
                    # pos_period=10_000,
                )
            elif params.training.cpc_loss.prediction_type == "recur":
                penc = RecurrentEncoder(
                    self.context.output_size,
                    penc_out_size,
                    recurrent_type="lstm",
                    dropout_prob=params.training.dropout_prob,
                )
            else:
                penc = FeedForwardEncoder(
                    self.context.output_size,
                    penc_out_size,
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
                self._num_speakers,
                params.training.dropout_prob,
                params.training.cpc_loss.offset,
                params.training.cpc_loss.gutted_steps,
            )
        elif params.training.loss_type == "best-rq":
            self.register_module("cpc_loss", None)
            if params.training.best_rq_loss is None:
                raise ValueError("params.training.best_rq_loss not initialized")
            self._num_speakers = params.training.best_rq_loss.num_speakers
            if self._num_speakers is not None:
                self._speaker_regex = re.compile(
                    params.training.best_rq_loss.speaker_regex
                )
                if self._speaker_regex.groups != 1:
                    raise ValueError(
                        f"expected one group in regex '{self._speaker_regex.pattern}'; "
                        f"got {self._speaker_regex.groups}"
                    )
            else:
                self._speaker_regex = None
            self.masker = ContiguousTemporalMask(
                params.training.best_rq_loss.mask_width,
                params.training.best_rq_loss.mask_prob,
                params.training.best_rq_loss.mask_mean,
                params.training.best_rq_loss.mask_std,
            )
            if params.training.chunking.policy != "fixed" and None in {
                params.training.best_rq_loss.mask_mean,
                params.training.best_rq_loss.mask_std,
            }:
                logger = logging.getLogger("pytorch_lightning")
                logger.warn(
                    f"chunking policy '{params.training.chunking.policy}' != 'fixed', "
                    "but BEST-RQ mask mean/stddev are being determined dynamically. "
                    "Computations may be incorrect due to right-padding"
                )
            if params.training.best_rq_loss.prediction_type == "csa":
                penc = CausalSelfAttentionEncoder(
                    self.context.output_size,
                    params.training.best_rq_loss.codebook_size,
                    dropout_prob=params.training.dropout_prob,
                )
            elif params.training.best_rq_loss.prediction_type == "sa":
                penc = SelfAttentionEncoder(
                    self.context.output_size,
                    params.training.best_rq_loss.codebook_size,
                    dropout_prob=params.training.dropout_prob,
                )
            elif params.training.best_rq_loss.prediction_type == "recur":
                penc = RecurrentEncoder(
                    self.context.output_size,
                    params.training.best_rq_loss.codebook_size,
                    recurrent_type="lstm",
                    dropout_prob=params.training.dropout_prob,
                )
            else:
                penc = FeedForwardEncoder(
                    self.context.output_size,
                    params.training.best_rq_loss.codebook_size,
                    "none",
                    False,
                    params.training.dropout_prob,
                )
            self.best_rq_loss = BestRqLossNetwork(
                params.input_size,
                penc,
                params.training.best_rq_loss.codebook_dim,
                self._num_speakers,
                params.training.best_rq_loss.offset,
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
        assert self._num_speakers is not None
        for n, utt_id in enumerate(utt_ids):
            speaker = self._speaker_regex.match(utt_id).group(1)
            id_ = self._speaker_map.get(speaker, None)
            if id_ is None:
                self._speaker_min += 1
                if self._speaker_min >= self._num_speakers:
                    logger = logging.getLogger("pytorch_lightning")
                    logger.warn(f"number of speakers exceeded {self._num_speakers}")
                    self._speaker_min = self._num_speakers - 1
                id_ = self._speaker_min
                self._speaker_map[speaker] = id_
            speakers[n] = id_
        return speakers

    def pretrain_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        feats, _, _, feat_sizes, _, utt_ids = batch
        if len(utt_ids) == 0:
            return feats.new_zeros(1)
        speakers = self.get_speakers_from_uttids(utt_ids).to(feats.device)
        if self.masker is not None:
            with torch.no_grad():
                feats_ = self.masker(feats)
        else:
            feats_ = feats
        latent, lens = self.latent(feats_, feat_sizes)
        context, lens = self.context(latent, lens)
        if self.params.training.loss_type == "cpc":
            loss = self.cpc_loss(latent, context, lens, speakers)
        elif self.params.training.loss_type == "best-rq":
            loss = self.best_rq_loss(feats, context, lens, speakers)
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


def main(args: Optional[Sequence[str]] = None):
    """Pre-train an scpc model"""
    parser = argparse.ArgumentParser(
        description=main.__doc__, fromfile_prefix_chars="@"
    )
    parser.add_argument("train_dir")
    parser.add_argument("val_dir")
    parser.add_argument(
        "best",
        nargs="?",
        default=None,
        help="Where to save best inference model chekpoint to. Defaults to "
        "'<root_dir>/<model_name>/version_<version>/best.ckpt' if --version is "
        "specified, otherwise '<root_dir>/<model_name>/best.ckpt'",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="What version/run of this model to perform/continue",
    )
    LightningPretrainedFrontend.add_argparse_args(
        parser,
        read_format_str="--read-model-{file_format}",
        print_format_str="--print-model-{file_format}",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Number of workers in datasets"
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        default=False,
        help="Suppress progress bar",
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        help="The root experiment directory. Defaults to current working directory",
    )

    options = parser.parse_args(args)

    lpf = LightningPretrainedFrontend.from_argparse_args(options)
    tparams = lpf.params.training
    dparams = tparams.data
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
        shuffle=tparams.shuffle,
        num_workers=options.num_workers,
        warn_on_missing=False,
        on_uneven_distributed="raise",
    )
    root_dir = options.root_dir
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
    enable_progress_bar = not options.no_progress_bar
    if enable_progress_bar:
        callbacks.append(RichProgressBar())
    logger_dir = os.path.join(root_dir, "tb_logs")
    os.makedirs(logger_dir, exist_ok=True)
    logger = TensorBoardLogger(logger_dir, model_name, options.version)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=root_dir,
        replace_sampler_ddp=False,
        accelerator=tparams.accelerator,
        strategy="ddp" if tparams.num_devices else None,
        devices=tparams.num_devices if tparams.num_devices else 1,
        num_nodes=tparams.num_nodes,
        max_epochs=tparams.max_epochs,
        enable_progress_bar=enable_progress_bar,
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


if __name__ == "__main__":
    sys.exit(main())
