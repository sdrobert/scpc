# Copyright 2023 Sean Robertson
#
# Much of this code is based on that of github.com/facebookresearch/cpc_audio, which is
# MIT-licensed. See LICENSE_cpc_audio for license details.
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

import abc
import os
from typing import (
    IO,
    Any,
    BinaryIO,
    Collection,
    Dict,
    Final,
    Generic,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Self

import torch
import numpy as np

__all__ = [
    "CausalSelfAttentionEncoder",
    "ConvEncoder",
    "CPCLossNetwork",
    "Encoder",
    "EncoderSequence",
    "FeedForwardEncoder",
    "IdentityEncoder",
    "RecurrentEncoder",
    "SelfAttentionEncoder",
]


@torch.no_grad()
def get_length_mask(
    x: torch.Tensor, lens: torch.Tensor, batch_dim: int = 0, seq_dim: int = -1
) -> torch.Tensor:
    N, T = x.size(batch_dim), x.size(seq_dim)
    assert lens.shape == (N,)
    arange_shape, lens_shape = [1] * x.ndim, [1] * x.ndim
    arange_shape[seq_dim] = T
    lens_shape[batch_dim] = N
    arange = torch.arange(T, device=x.device).view(arange_shape)
    lens = lens.view(lens_shape)
    return (lens > arange).expand_as(x)


def check_positive(name: str, val, nonnegative=False):
    pos = "non-negative" if nonnegative else "positive"
    if val < 0 or (val == 0 and not nonnegative):
        raise ValueError(f"Expected {name} to be {pos}; got {val}")


def check_in(name: str, val: str, choices: Collection[str]):
    if val not in choices:
        choices = "', '".join(sorted(choices))
        raise ValueError(f"Expected {name} to be one of '{choices}'; got '{val}'")


T = TypeVar("T")


def check_is_instance(name: str, val: T, cls: Type[T]):
    if not isinstance(val, cls):
        raise ValueError(
            f"Expected {name} to be a '{cls.__name__}'; got '{type(val).__name__}'"
        )


def check_equals(name: str, val: Any, other: Any):
    if val != other:
        raise ValueError(f"Expected {name} to be equal to '{other}'; got '{val}'")


class MaskingLayer(torch.nn.Module):
    __slots__ = "window", "padding", "stride", "batch_dim", "seq_dim"
    window: int
    stride: int
    padding: int
    batch_dim: int
    seq_dim: int

    def __init__(
        self,
        window: int = 1,
        stride: int = 1,
        padding: int = 0,
        batch_dim: int = 0,
        seq_dim: int = -1,
    ) -> None:
        check_positive("window", window)
        check_positive("stride", stride)
        check_positive("padding", padding, True)
        super().__init__()
        self.window = window
        self.stride = stride
        self.padding = padding
        self.batch_dim = batch_dim
        self.seq_dim = seq_dim

    def forward(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if lens is None:
            return x, lens
        lens = (lens + 2 * self.padding - self.window) // self.stride + 1
        return (
            x.masked_fill(~get_length_mask(x, lens, self.batch_dim, self.seq_dim), 0.0),
            lens,
        )


class ChannelNorm(torch.nn.Module):
    __slots__ = "epsilon"

    epsilon: float

    def __init__(self, num_features: int, affine: bool = True, epsilon: float = 1e-05):
        check_positive("num_features", num_features)
        check_positive("epsilon", epsilon)
        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = torch.nn.parameter.Parameter(torch.empty(1, num_features, 1))
            self.bias = torch.nn.parameter.Parameter(torch.empty(1, num_features, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.epsilon = epsilon
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.epsilon)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


E = TypeVar("E", bound="Encoder", covariant=True)


class Encoder(torch.nn.Module, metaclass=abc.ABCMeta):
    __slots__ = "input_size", "output_size"
    input_size: int
    output_size: int

    JSON_NAME: str
    JSON_NAME2ENC: Dict[str, Type[E]] = dict()
    JSON_STATE_DICT_ENTRY: Final = "json"

    def __init__(self, input_size: int, output_size: Optional[int] = None) -> None:
        check_positive("input_size", input_size)
        if output_size is None:
            output_size = input_size
        else:
            check_positive("output_size", output_size)
        super().__init__()
        self.input_size, self.output_size = input_size, output_size

    @abc.abstractproperty
    def downsampling_factor(self) -> int:
        ...

    @abc.abstractmethod
    def encode(
        self, x: torch.Tensor, lens: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...

    def forward(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, lens = self.encode(x, lens)
        if lens is not None:
            x = x.masked_fill(~get_length_mask(x, lens, seq_dim=1), 0.0)
        return x, lens

    @classmethod
    def __init_subclass__(cls, /, json_name: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if json_name in Encoder.JSON_NAME2ENC:
            raise ValueError(
                f"Cannot Encoder with json_name '{json_name}'; already registered"
            )
        cls.JSON_NAME = json_name
        Encoder.JSON_NAME2ENC[json_name] = cls

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.JSON_NAME, "args": [self.input_size, self.output_size]}

    @classmethod
    def from_json(cls, json: Dict[str, Any]) -> Self:
        json_name = json["name"]
        if not isinstance(json_name, str):
            raise ValueError("json value of 'name' is not a string")
        cls_ = Encoder.JSON_NAME2ENC[json_name]
        if cls is cls_:
            args = json.get("args", tuple())
            kwargs = json.get("kwargs", dict())
            return cls(*args, **kwargs)
        elif issubclass(cls_, cls):
            return cls_.from_json(json)
        else:
            raise ValueError(
                f"json 'name' key had value '{json_name}', but '{cls_.__name__}' "
                f"is not a subclass of '{cls.__name__}'"
            )

    def save_checkpoint(self, f: Union[str, BinaryIO, IO[bytes], os.PathLike]):
        state_dict = self.state_dict()
        assert self.JSON_STATE_DICT_ENTRY not in state_dict
        state_dict[self.JSON_STATE_DICT_ENTRY] = self.to_json()
        torch.save(state_dict, f)

    @classmethod
    def from_checkpoint(
        cls, f: Union[str, BinaryIO, IO[bytes], os.PathLike], strict: bool = True
    ) -> Self:
        state_dict = torch.load(f)
        if not isinstance(state_dict, dict):
            raise ValueError(f"checkpoint file does not contain a dictionary")
        elif cls.JSON_STATE_DICT_ENTRY not in state_dict:
            raise ValueError(
                f"checkpoint file contains no entry '{cls.JSON_STATE_DICT_ENTRY}' "
                "(did you save this with save_checkpoint?)"
            )
        json = state_dict.pop(cls.JSON_STATE_DICT_ENTRY)
        encoder = cls.from_json(json)
        encoder.load_state_dict(state_dict, strict)
        return encoder


class IdentityEncoder(Encoder, json_name="id"):
    @property
    def downsampling_factor(self) -> int:
        return 1

    def encode(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return x, lens


NonlinearityType = Literal["relu", "sigmoid", "tanh", "none"]


class FeedForwardEncoder(Encoder, json_name="ff"):
    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        nonlin_type: NonlinearityType = "relu",
        bias: bool = True,
        dropout_prob: float = 0.0,
    ) -> None:
        check_positive("dropout_prob", dropout_prob, True)
        check_in("nonlin_type", nonlin_type, {"relu", "sigmoid", "tanh", "none"})
        super().__init__(input_size, output_size)
        self.ff = torch.nn.Linear(self.input_size, self.output_size, bias)
        self.drop = torch.nn.Dropout(dropout_prob)
        if nonlin_type == "relu":
            self.nonlin = torch.nn.ReLU()
        elif nonlin_type == "sigmoid":
            self.nonlin = torch.nn.Sigmoid()
        elif nonlin_type == "tanh":
            self.nonlin = torch.nn.Tanh()
        else:
            self.nonlin = torch.nn.Identity()

    @property
    def nonlin_type(self) -> NonlinearityType:
        if isinstance(self.nonlin, torch.nn.ReLU):
            return "relu"
        elif isinstance(self.nonlin, torch.nn.Sigmoid):
            return "sigmoid"
        elif isinstance(self.nonlin, torch.nn.Tanh):
            return "tanh"
        else:
            return "none"

    @property
    def bias(self) -> bool:
        return self.ff.bias is not None

    @property
    def dropout_prob(self) -> float:
        return self.drop.p

    def to_json(self) -> Dict[str, Any]:
        dict_ = super().to_json()
        assert isinstance(dict_["args"], list)
        dict_["args"].extend([self.nonlin_type, self.bias, self.dropout_prob])
        return dict_

    @property
    def downsampling_factor(self) -> int:
        return 1

    def encode(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.nonlin(self.ff(self.drop(x)))
        return x, lens


NormStyle = Literal["none", "batch", "instance", "channel"]


class ConvEncoder(Encoder, json_name="conv"):
    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        norm_style: NormStyle = "none",
        dropout_prob: float = 0.0,
    ) -> None:
        check_positive("dropout_prob", dropout_prob, True)
        check_in("norm_style", norm_style, {"none", "batch", "instance", "channel"})
        super().__init__(input_size, output_size)
        input_size, output_size = self.input_size, self.output_size
        if norm_style == "none":
            Norm = lambda: torch.nn.Identity()
        elif norm_style == "batch":
            Norm = lambda: torch.nn.BatchNorm1d(output_size)
        elif norm_style == "instance":
            Norm = lambda: torch.nn.InstanceNorm1d(
                output_size, track_running_stats=True
            )
        else:
            Norm = lambda: ChannelNorm(output_size)

        self.drop = torch.nn.Dropout(dropout_prob)
        self.relu = torch.nn.ReLU()
        self.mask0 = MaskingLayer(1, 1, 0)
        self.conv1 = torch.nn.Conv1d(input_size, output_size, 10, 5, 3)
        self.norm1 = Norm()
        self.mask1 = MaskingLayer(10, 5, 3)
        self.conv2 = torch.nn.Conv1d(output_size, output_size, 8, 4, 2)
        self.norm2 = Norm()
        self.mask2 = MaskingLayer(8, 4, 2)
        self.conv3 = torch.nn.Conv1d(output_size, output_size, 4, 2, 1)
        self.norm3 = Norm()
        self.mask3 = MaskingLayer(4, 2, 1)
        self.conv4 = torch.nn.Conv1d(output_size, output_size, 4, 2, 1)
        self.norm4 = Norm()
        self.conv5 = torch.nn.Conv1d(output_size, output_size, 4, 2, 1)
        self.norm5 = Norm()

    @property
    def downsampling_factor(self) -> int:
        return 160

    @property
    def norm_style(self) -> NormStyle:
        if isinstance(self.norm1, torch.nn.BatchNorm1d):
            return "batch"
        elif isinstance(self.norm1, torch.nn.InstanceNorm1d):
            return "instance"
        elif isinstance(self.norm1, ChannelNorm):
            return "channel"
        else:
            return "none"

    @property
    def dropout_prob(self) -> float:
        return self.drop.p

    def to_json(self) -> Dict[str, Any]:
        dict_ = super().to_json()
        assert isinstance(dict_["args"], list)
        dict_["args"].extend([self.norm_style, self.dropout_prob])
        return dict_

    def encode(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x.transpose(1, 2)  # N, C, T
        x, lens = self.mask0(x, lens)
        x, lens = self.mask1(self.relu(self.norm1(self.conv1(self.drop(x)))), lens)
        x, lens = self.mask2(self.relu(self.norm2(self.conv2(self.drop(x)))), lens)
        x, lens = self.mask3(self.relu(self.norm3(self.conv3(self.drop(x)))), lens)
        x, lens = self.mask3(self.relu(self.norm4(self.conv4(self.drop(x)))), lens)
        x, lens = self.mask3(self.relu(self.norm5(self.conv5(self.drop(x)))), lens)
        x = x.transpose(1, 2)
        return x, lens

    def forward(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # already padded
        return self.encode(x, lens)


class SelfAttentionEncoder(Encoder, json_name="sa"):
    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout_prob: float = 0.0,
        enable_nested_tensor: bool = False,
    ) -> None:
        check_positive("num_layers", num_layers)
        check_positive("num_heads", num_heads)
        check_positive("dim_feedforward", dim_feedforward)
        check_positive("dropout_prob", dropout_prob, True)
        super().__init__(input_size, output_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            input_size,
            num_heads,
            dim_feedforward,
            dropout_prob,
            "relu",
            batch_first=True,
        )
        layer_norm = torch.nn.LayerNorm(input_size)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers, layer_norm, enable_nested_tensor
        )
        if output_size is None:
            self.ff = torch.nn.Identity()
        else:
            self.ff = torch.nn.Linear(input_size, output_size)

    @property
    def num_layers(self) -> int:
        return self.encoder.num_layers

    @property
    def encoder_layer(self) -> torch.nn.TransformerEncoderLayer:
        assert len(self.encoder.layers)
        layer = self.encoder.layers[0]
        assert isinstance(layer, torch.nn.TransformerEncoderLayer)
        return layer

    @property
    def num_heads(self) -> int:
        return self.encoder_layer.self_attn.num_heads

    @property
    def dim_feedforward(self) -> int:
        return self.encoder_layer.linear1.out_features

    @property
    def dropout_prob(self) -> int:
        return self.encoder_layer.dropout1.p

    @property
    def enable_nested_tensor(self) -> bool:
        return self.encoder.enable_nested_tensor

    @property
    def downsampling_factor(self) -> int:
        return 1

    def to_json(self) -> Dict[str, Any]:
        dict_ = super().to_json()
        assert isinstance(dict_["args"], list)
        if isinstance(self.ff, torch.nn.Identity):
            dict_["args"][-1] = None
        dict_["args"].extend(
            [
                self.num_layers,
                self.num_heads,
                self.dim_feedforward,
                self.dropout_prob,
                self.enable_nested_tensor,
            ]
        )
        return dict_

    def get_mask(self, x: torch.Tensor, lens: Optional[torch.Tensor]) -> torch.Tensor:
        N, T = x.shape[:2]
        out_shape = (self.num_heads * N, T, T)
        len_mask = x.new_ones(out_shape, dtype=torch.bool)
        if lens is not None:
            len_mask = get_length_mask(len_mask, lens.repeat_interleave(self.num_heads))
        return len_mask

    def encode(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mask = ~self.get_mask(x, lens)
        x = self.ff(self.encoder(x, mask))
        return x, lens


class CausalSelfAttentionEncoder(SelfAttentionEncoder, json_name="csa"):
    __slots__ = "max_width"
    max_width: Optional[int]

    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        max_width: Optional[int] = None,
        num_layers: int = 1,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout_prob: float = 0.0,
        enable_nested_tensor: bool = False,
    ) -> None:
        if max_width is not None:
            check_positive("max_width", max_width)
        super().__init__(
            input_size,
            output_size,
            num_layers,
            num_heads,
            dim_feedforward,
            dropout_prob,
            enable_nested_tensor,
        )
        self.max_width = max_width

    def to_json(self) -> Dict[str, Any]:
        dict_ = super().to_json()
        assert isinstance(dict_["args"], list)
        dict_["args"].insert(2, self.max_width)
        return dict_

    def get_mask(self, x: torch.Tensor, lens: Optional[torch.Tensor]) -> torch.Tensor:
        len_mask = super().get_mask(x, lens)
        T = len_mask.size(-1)
        causal_mask = len_mask.new_ones(T, T).tril_()
        if self.max_width is not None:
            causal_mask = causal_mask.triu_(-self.max_width + 1)
        return len_mask & causal_mask


RecurrentType = Literal["gru", "lstm", "rnn"]


class RecurrentEncoder(Encoder, json_name="ra"):
    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        recurrent_type: RecurrentType = "gru",
        dropout_prob: float = 0.0,
    ) -> None:
        check_positive("num_layers", num_layers)
        check_in("recurrent_type", recurrent_type, {"gru", "lstm", "rnn"})
        check_positive("dropout_prob", dropout_prob, True)
        super().__init__(input_size, output_size)
        if recurrent_type == "gru":
            RNN = torch.nn.GRU
        elif recurrent_type == "lstm":
            RNN = torch.nn.LSTM
        else:
            RNN = torch.nn.RNN
        self.rnn = RNN(
            self.input_size,
            self.output_size,
            num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout_prob,
        )

    @property
    def num_layers(self) -> int:
        return self.rnn.num_layers

    @property
    def recurrent_type(self) -> RecurrentType:
        if isinstance(self.rnn, torch.nn.GRU):
            return "gru"
        elif isinstance(self.rnn, torch.nn.LSTM):
            return "lstm"
        else:
            return "rnn"

    @property
    def dropout_prob(self) -> float:
        return self.rnn.dropout

    @property
    def downsampling_factor(self) -> int:
        return 1

    def to_json(self) -> Dict[str, Any]:
        dict_ = super().to_json()
        assert isinstance(dict_["args"], list)
        dict_["args"].extend([self.num_layers, self.recurrent_type, self.dropout_prob])
        return dict_

    def encode(
        self, x: torch.Tensor, lens: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        T = x.size(1)
        if lens is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), True, False)
        x = self.rnn(x)[0]
        if lens is not None:
            x = torch.nn.utils.rnn.pad_packed_sequence(x, True, 0, T)[0]
        return x, lens

    def forward(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # already padded
        return self.encode(x, lens)


class EncoderSequence(Encoder, json_name="seq"):
    encoders: Sequence[Encoder]

    @overload
    def __init__(self, *encoders: Encoder) -> None:
        ...

    def __init__(self, encoder: Encoder, *encoders: Encoder) -> None:
        input_size, output_size = encoder.input_size, encoder.output_size
        for i, e in enumerate(encoders):
            check_equals(f"encoders[{i+1}]", e.input_size, output_size)
            output_size = e.output_size
        super().__init__(input_size, output_size)
        self.encoders = torch.nn.ModuleList((encoder,) + encoders)

    @property
    def downsampling_factor(self) -> int:
        return np.prod(e.downsampling_factor for e in self.encoders)

    def to_json(self) -> Dict[str, Any]:
        dict_ = super().to_json()
        dict_["args"] = [e.to_json() for e in self.encoders]
        return dict_

    @classmethod
    def from_json(cls, json: Dict[str, Any]) -> Self:
        json_name = json["name"]
        if not isinstance(json_name, str):
            raise ValueError("json value of 'name' is not a string")
        cls_ = Encoder.JSON_NAME2ENC[json_name]
        if cls is cls_:
            encoders = (Encoder.from_json(x) for x in json["args"])
            return cls(*encoders)
        else:
            return super().from_json(json)

    def encode(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for e in self.encoders:
            x, lens = e(x, lens)
        return x, lens

    def forward(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # already padded
        return self.encode(x, lens)


class CPCLossNetwork(torch.nn.Module, Generic[E]):
    __slots__ = "negative_samples", "prediction_steps"

    negative_samples: int
    prediction_steps: int
    prediction_encoder: E

    @overload
    def __init__(
        self: "CPCLossNetwork[FeedForwardEncoder]",
        latent_size: int,
        context_size: Optional[int] = None,
        prediction_steps: int = 12,
        negative_samples: int = 128,
        predictionEncoder: Literal[None] = None,
        num_speakers: Optional[int] = None,
        dropout_prob: float = 0.0,
    ):
        ...

    @overload
    def __init__(
        self: "CPCLossNetwork[E]",
        latent_size: int,
        context_size: Optional[int] = None,
        prediction_steps: int = 12,
        negative_samples: int = 128,
        predictionEncoder: E = None,
        num_speakers: Optional[int] = None,
        dropout_prob: float = 0.0,
    ):
        ...

    def __init__(
        self,
        latent_size: int,
        context_size: Optional[int] = None,
        prediction_steps: int = 12,
        negative_samples: int = 128,
        prediction_encoder: Optional[E] = None,
        num_speakers: Optional[int] = None,
        dropout_prob: float = 0.0,
    ) -> None:
        check_positive("latent_size", latent_size)
        if context_size is None:
            context_size = latent_size
        else:
            check_positive("context_size", context_size)
        check_positive("prediction_steps", prediction_steps)
        check_positive("negative_samples", negative_samples)
        check_positive("dropout_prob", dropout_prob, True)
        if num_speakers is not None:
            check_positive("num_speakers", num_speakers)
        if prediction_encoder is None:
            prediction_encoder = FeedForwardEncoder(
                context_size,
                prediction_steps * latent_size,
                "none",
                False,
                dropout_prob=dropout_prob,
            )
        else:
            check_is_instance("prediction_encoder", prediction_encoder, Encoder)
            check_equals(
                "prediction_encoder.input_size",
                prediction_encoder.input_size,
                context_size,
            )
            check_equals(
                "prediction_encoder.output_size",
                prediction_encoder.output_size,
                prediction_steps * latent_size,
            )

        super().__init__()
        self.negative_samples = negative_samples
        self.prediction_steps = prediction_steps

        if num_speakers is not None:
            self.embed = torch.nn.Embedding(num_speakers, context_size)
        else:
            self.register_module("embed", None)

        self.prediction_encoder = prediction_encoder

        self.unfold = torch.nn.Unfold((prediction_steps, latent_size))

    def forward(
        self,
        latent: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        lens: Optional[torch.Tensor] = None,
        sources: Optional[torch.Tensor] = None,
    ):
        if context is None:
            context = latent

        N, T, C = latent.shape
        assert N > 0 and T > 0, (N, T)
        assert context.shape[:2] == (N, T)
        K, M = self.prediction_steps, self.negative_samples
        Tp = T - self.prediction_steps
        assert Tp > 0, "prediction window too large for sequences"

        if self.embed is not None:
            context = context + self.embed(sources).unsqueeze(1)

        lens_ = None if lens is None else lens.clamp_max(Tp)
        Az = self.prediction_encoder(context[:, :Tp], lens_)[0].view(N * Tp * K, C, 1)

        if lens is None:
            phi_n = latent.flatten(end_dim=1)
            norm = latent.new_ones(1)
        else:
            norm = (lens - K).clamp_min_(0).sum() * K
            assert norm > 0, "prediction window too large"
            mask = get_length_mask(latent, lens, seq_dim=1)
            phi_n = latent[mask].view(-1, latent.size(2))
        samps = torch.randint(phi_n.size(0), (N * Tp * M,), device=latent.device)
        phi_n = phi_n[samps].view(N * Tp, 1, M, C).expand(N * Tp, K, M, C).flatten(0, 1)
        denom = torch.bmm(phi_n, Az).squeeze(2)
        denom = denom.logsumexp(1).view(N, Tp, K)
        del phi_n

        phi_k = self.unfold(latent[:, 1:].unsqueeze(1))  # (N, K * C, Tp)
        phi_k = phi_k.transpose(1, 2).reshape(N * Tp * K, 1, C)
        num = torch.bmm(phi_k, Az).view(N, Tp, K)
        denom = num.logaddexp(denom)
        del phi_k

        loss = denom - num  # neg num - denom
        if lens is None:
            loss = loss.mean()
        else:
            mask = get_length_mask(loss, lens - K, seq_dim=1)
            loss = loss.masked_fill(~mask, 0.0).sum()
            loss = loss / norm
        return loss  # + math.log(M + 1)
