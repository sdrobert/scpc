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

from typing import Literal, Optional

import torch
import param
import pytorch_lightning as pl

from pydrobert.torch.modules import ChunkBySlices, SliceSpectData

from .modules import *

__all__ = [
    "PretrainedFrontend",
    "PretrainedFrontendParams",
]


class ConvEncoderParams(param.Parameterized):

    output_size: int = param.Integer(
        256, bounds=(1, None), doc="Size of output (and # of conv output channels)"
    )
    channel_norm_eps: float = param.Number(
        1e-5,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="Minimum denominator value in channel norm",
    )


class SelfAttentionEncoderParams(param.Parameterized):

    num_layers: int = param.Integer(
        1, bounds=(1, None), doc="Number of self-attention layers"
    )
    num_heads: int = param.Integer(1, bounds=(1, None), doc="Number of attention heads")
    dim_feedforward: int = param.Integer(
        1, bounds=(1, None), doc="Size of intermediate representation"
    )
    layer_norm_eps: float = param.Number(
        1e-5,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        doc="Minimum denominator value in layer norm",
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
    num_sources: int = param.Integer(
        0,
        bounds=(0, None),
        doc="Number of sources (i.e. speakers) to construct embeddings for. "
        "0 means no embeddings will be used",
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


class TrainingParams(param.Parameterized):

    optimizer: Literal["adam", "sgd"] = param.ObjectSelector(
        "adam", ["adam", "sgd"], doc="Optimizer class"
    )
    learning_rate: float = param.Number(
        2e-4, bounds=(0, None), inclusive_bounds=(False, True), doc="Learning rate"
    )
    batch_size: int = param.Integer(
        8, bounds=(1, None), doc="Batch size (pre-chunking)"
    )
    dropout_prob: float = param.Magnitude(0.1, doc="Probability of dropping out a unit")

    chunking: ChunkingParams = param.ClassSelector(
        CPCLossParams, doc="Parameters for chunking"
    )

    loss_type: Literal["cpc"] = param.ObjectSelector(
        "cpc", ["cpc"], doc="Loss to train model with"
    )
    cpc_loss: CPCLossParams = param.ClassSelector(
        CPCLossParams, doc="Parameters for CPC loss (if loss_type = 'cpc')"
    )


class PretrainedFrontendParams(param.Parameterized):

    input_size: int = param.Integer(
        1, bounds=(1, None), doc="Size of input feature dimension (1 for raw)"
    )

    latent_type: Literal["conv", "id"] = param.ObjectSelector(
        "conv",
        ["conv", "id"],
        doc="Which encoder to use for the 'latent' part of the model. 'conv' is "
        "convolutional; 'id' is identity (noop)",
    )
    context_type: Literal["csa", "sa", "recur", "id"] = param.ObjectSelector(
        "csa",
        ["csa", "sa", "recur", "id"],
        doc="Which encoder to use for the 'context' part of the model. 'csa' is "
        "causal self-atttention; 'sa' is self-attention (non-causal); 'recur' is "
        "recurrent; 'id' is identity (noop)",
    )

    conv: ConvEncoderParams = param.ClassSelector(
        ConvEncoderParams,
        doc="Parameters for latent convolutional encoder (if latent_type = 'conv')",
    )
    csa: CausalSelfAttentionEncoderParams = param.ClassSelector(
        CausalSelfAttentionEncoderParams,
        doc="Parameters for context causal self-attention encoder "
        "(if context_type = 'csa')",
    )
    sa: SelfAttentionEncoderParams = param.ClassSelector(
        SelfAttentionEncoderParams,
        doc="Parameters for context self-attention encoder (if context_type = 'sa')",
    )
    recur: RecurrentEncoderParams = param.ClassSelector(
        RecurrentEncoderParams,
        doc="Parameters for context recurrent encoder (if context_type = 'recur')",
    )

    training: TrainingParams = param.ClassSelector(
        TrainingParams, doc="Parameters for training"
    )


class PretrainedFrontend(pl.LightningModule):

    latent: Encoder
    context: Encoder
    slicer: Optional[SliceSpectData]
    chunker: Optional[ChunkBySlices]

    def __init__(self, params: PretrainedFrontendParams) -> None:
        super().__init__()

        if params.latent_type == "conv":
            self.latent = ConvEncoder(
                params.input_size,
                params.conv.output_size,
                params.conv.channel_norm_eps,
                params.training.dropout_prob,
            )
        elif params.latent_type == "id":
            self.latent = IdentityEncoder(params.input_size)
        else:
            raise NotImplementedError

        if params.context_type == "csa":
            self.context = CausalSelfAttentionEncoder(
                self.latent.output_size,
                params.csa.max_width,
                params.csa.num_layers,
                params.csa.num_heads,
                params.csa.dim_feedforward,
                params.csa.layer_norm_eps,
                params.training.dropout_prob,
                params.training.chunking.policy != "fixed",
            )
        elif params.context_type == "sa":
            self.context = SelfAttentionEncoder(
                self.latent.output_size,
                params.sa.num_layers,
                params.sa.num_heads,
                params.sa.dim_feedforward,
                params.sa.layer_norm_eps,
                params.training.dropout_prob,
                params.training.chunking.policy != "fixed",
            )
        elif params.context_type == "recur":
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
            self.cpc_loss = CPCLossNetwork(
                self.latent.output_size,
                self.context.output_size,
                params.training.cpc_loss.prediction_steps,
                params.training.cpc_loss.negative_samples,
                params.training.cpc_loss if params.training.cpc_loss else None,
                params.training.dropout_prob,
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

        self.save_hyperparameters()

    @property
    def params(self) -> PretrainedFrontendParams:
        return self.hparams.params

    @property
    def training_is_fixed_width(self) -> bool:
        return self.params.training.chunking.policy == "fixed"

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.params.training.optimizer == "adam":
            Optim = torch.optim.Adam
        elif self.params.training.optimizer == "sgd":
            Optim = torch.optim.SGD
        else:
            raise NotImplementedError

        return Optim(self.parameters(), lr=self.params.training.learning_rate)
