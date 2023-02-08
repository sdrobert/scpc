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
from typing import Collection, Literal, Optional, Tuple, Type

import torch

try:
    import pytest

    _parametrize = pytest.mark.parametrize
except ImportError:
    _parametrize = lambda x: x


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


class MaskingLayer(torch.nn.Module):

    __slots__ = ["window", "padding", "stride", "batch_dim", "seq_dim"]
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

    __slots__ = ["epsilon"]

    epsilon: float

    def __init__(self, num_features: int, epsilon: float = 1e-05, affine: bool = True):
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


class Encoder(torch.nn.Module, metaclass=abc.ABCMeta):

    __slots__ = ["input_size", "output_size"]
    input_size: int
    output_size: int

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


class IdentityEncoder(Encoder):
    @property
    def downsampling_factor(self) -> int:
        return 1

    def encode(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return x, lens


class ConvEncoder(Encoder):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        channel_norm_epsilon: float = 1e-5,
        dropout_prob: float = 0.1,
    ) -> None:
        check_positive("dropout_prob", dropout_prob)
        super().__init__(input_size, output_size)
        self.drop = torch.nn.Dropout(dropout_prob)
        self.relu = torch.nn.ReLU()
        self.mask0 = MaskingLayer(1, 1, 0)
        self.conv1 = torch.nn.Conv1d(input_size, output_size, 10, 5, 3)
        self.norm1 = ChannelNorm(output_size, channel_norm_epsilon)
        self.mask1 = MaskingLayer(10, 5, 3)
        self.conv2 = torch.nn.Conv1d(output_size, output_size, 8, 4, 2)
        self.norm2 = ChannelNorm(output_size, channel_norm_epsilon)
        self.mask2 = MaskingLayer(8, 4, 2)
        self.conv3 = torch.nn.Conv1d(output_size, output_size, 4, 2, 1)
        self.norm3 = ChannelNorm(output_size, channel_norm_epsilon)
        self.mask3 = MaskingLayer(4, 2, 1)
        self.conv4 = torch.nn.Conv1d(output_size, output_size, 4, 2, 1)
        self.norm4 = ChannelNorm(output_size, channel_norm_epsilon)
        self.conv5 = torch.nn.Conv1d(output_size, output_size, 4, 2, 1)
        self.norm5 = ChannelNorm(output_size, channel_norm_epsilon)

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.norm1.reset_parameters()
        self.conv2.reset_parameters()
        self.norm2.reset_parameters()
        self.conv3.reset_parameters()
        self.norm3.reset_parameters()
        self.conv4.reset_parameters()
        self.norm4.reset_parameters()
        self.conv5.reset_parameters()
        self.norm5.reset_parameters()

    @property
    def downsampling_factor(self) -> int:
        return 160

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
        x = x.transpose(1, 2)  # N, T, C
        return x, lens


class TransformerEncoder(Encoder):

    __slots__ = ["input_size", "output_size", "num_heads"]
    num_heads: int

    def __init__(
        self,
        input_size: int,
        num_layers: int = 1,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        layer_norm_eps: float = 1e-5,
        dropout_prob: float = 0.1,
        enable_nested_tensor: bool = False,
    ) -> None:
        check_positive("num_layers", num_layers)
        check_positive("num_heads", num_heads)
        check_positive("dim_feedforward", dim_feedforward)
        check_positive("layer_norm_eps", layer_norm_eps)
        check_positive("dropout_prob", dropout_prob)
        super().__init__(input_size)
        self.num_heads = num_heads
        encoder_layer = torch.nn.TransformerEncoderLayer(
            input_size,
            num_heads,
            dim_feedforward,
            dropout_prob,
            "relu",
            layer_norm_eps,
            True,
        )
        layer_norm = torch.nn.LayerNorm(input_size, layer_norm_eps)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers, layer_norm, enable_nested_tensor
        )

    @property
    def downsampling_factor(self) -> int:
        return 1

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
        x = self.encoder(x, mask)
        return x, lens


class CausalTransformerEncoder(TransformerEncoder):

    __slots__ = ["input_size", "output_size", "num_heads", "max_width"]
    max_width: Optional[int]

    def __init__(
        self,
        input_size: int,
        max_width: Optional[int] = None,
        num_layers: int = 1,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        layer_norm_eps: float = 1e-5,
        dropout_prob: float = 0.1,
        enable_nested_tensor: bool = False,
    ) -> None:
        if max_width is not None:
            check_positive("max_width", max_width)
        super().__init__(
            input_size,
            num_layers,
            num_heads,
            dim_feedforward,
            layer_norm_eps,
            dropout_prob,
            enable_nested_tensor,
        )
        self.max_width = max_width

    def get_mask(self, x: torch.Tensor, lens: Optional[torch.Tensor]) -> torch.Tensor:
        len_mask = super().get_mask(x, lens)
        T = len_mask.size(-1)
        causal_mask = len_mask.new_ones(T, T).tril_()
        if self.max_width is not None:
            causal_mask = causal_mask.triu_(-self.max_width + 1)
        return len_mask & causal_mask


class RecurrentEncoder(Encoder):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        num_layers: int = 1,
        recurrent_type: Literal["gru", "lstm", "rnn"] = "gru",
        dropout_prob: float = 0.1,
    ) -> None:
        check_positive("num_layers", num_layers)
        check_in("recurrent_type", recurrent_type, {"gru", "lstm", "rnn"})
        check_positive("dropout_prob", dropout_prob)
        super().__init__(input_size, output_size)
        if recurrent_type == "gru":
            RNN = torch.nn.GRU
        elif recurrent_type == "lstm":
            RNN = torch.nn.LSTM
        else:
            RNN = torch.nn.RNN
        self.rnn = RNN(
            input_size,
            output_size,
            num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout_prob,
        )

    def downsampling_factor(self) -> int:
        return 1

    def encode(
        self, x: torch.Tensor, lens: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        T = x.size(1)
        if lens is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lens, True, False)
        x = self.rnn(x)[0]
        if lens is not None:
            x = torch.nn.utils.rnn.pad_packed_sequence(x, True, 0, T)[0]
        return x, lens


class CPCLossNetwork(torch.nn.Module):

    __slots__ = ["negative_samples", "prediction_steps"]

    negative_samples: int
    prediction_steps: int

    def __init__(
        self,
        latent_size: int,
        context_size: Optional[int] = None,
        prediction_steps: int = 12,
        negative_samples: int = 128,
        num_sources: Optional[int] = None,
        dropout_prob: float = 0.1,
    ) -> None:
        check_positive("latent_size", latent_size)
        if context_size is None:
            context_size = latent_size
        else:
            check_positive("context_size", context_size)
        check_positive("prediction_steps", prediction_steps)
        check_positive("negative_samples", negative_samples)
        if num_sources is not None:
            check_positive("num_sources", num_sources)
        check_positive("dropout_prob", dropout_prob)
        super().__init__()
        self.negative_samples = negative_samples
        self.prediction_steps = prediction_steps

        if num_sources is not None:
            self.embed = torch.nn.Embedding(num_sources, context_size)
        else:
            self.register_module("embed", None)

        self.ff = torch.nn.Linear(context_size, prediction_steps * latent_size, False)

        self.drop = torch.nn.Dropout(dropout_prob)

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
        assert N > 0 and T > 0
        assert context.shape[:2] == (N, T)
        K, M = self.prediction_steps, self.negative_samples
        Tp = T - self.prediction_steps
        assert Tp > 0, "prediction window too large for sequences"

        if self.embed is not None:
            context = context + self.embed(sources).unsqueeze(1)

        Az = self.drop(self.ff(context[:, :Tp])).view(N * Tp * K, C, 1)

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
        del phi_k

        loss = denom - num  # neg num - denom
        if lens is None:
            loss = loss.mean()
        else:
            mask = get_length_mask(loss, lens - K, seq_dim=1)
            loss = loss.masked_fill(~mask, 0.0).sum()
            loss = loss / norm
        return loss


## TESTS


@_parametrize(
    "Encoder",
    [
        IdentityEncoder,
        ConvEncoder,
        TransformerEncoder,
        CausalTransformerEncoder,
        RecurrentEncoder,
    ],
    ids=[
        "IdentityEncoder",
        "ConvEncoder",
        "TransformerEncoder",
        "CausalTransformerEncoder",
        "RecurrentEncoder",
    ],
)
def test_encoder_variable_batches(Encoder: Type[Encoder]):
    torch.manual_seed(0)
    N, Tmin, Tmax, H = 10, 300, 600, 256
    encoder = Encoder(H)
    encoder.eval()
    x, lens = [], []
    x_exp = []
    for _ in range(N):
        lens_n = torch.randint(Tmin, Tmax + 1, (1,))
        x_n = torch.rand(1, lens_n.item(), H)
        x_exp_n, lens__n = encoder(x_n)
        assert x_exp_n.size(-1) == encoder.output_size
        assert not (x_exp_n == 0).all()  # check we're not just padding nothing
        assert lens__n is None
        x.append(x_n.flatten())
        lens.append(lens_n)
        x_exp.append(x_exp_n.squeeze(0))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=-1.0)
    x = x.view(N, -1, H)
    lens = torch.cat(lens)
    x_act, lens_act = encoder(x, lens)
    for n, x_exp_n, x_act_n, lens_act_n in zip(range(N), x_exp, x_act, lens_act):
        assert lens_act_n == x_exp_n.size(0), n
        assert (x_act_n[lens_act_n:] == 0).all(), n
        assert torch.allclose(x_exp_n, x_act_n[:lens_act_n], atol=1e-5), n


def test_causal_transformer_encoder_is_causal():
    torch.manual_seed(1)
    N, T, H = 3, 13, 17
    x = torch.rand(N, T, H)
    encoder = CausalTransformerEncoder(H, T // 2 - 1, num_heads=1)
    encoder.eval()
    x1 = encoder(x)[0][:, : T // 2]
    x2 = encoder(x[:, : T // 2])[0]
    assert x1.shape == x2.shape
    assert torch.allclose(x1, x2, atol=1e-5)
    x1 = x1[:, -2:]
    x2 = encoder(x[:, 1 : T // 2])[0][:, -2:]
    assert not torch.allclose(x1[:, 0], x2[:, 0], atol=1e-5)
    assert torch.allclose(x1[:, 1], x2[:, 1], atol=1e-5)


@_parametrize("embedding", [True, False], ids=["w/-embedding", "w/o-embedding"])
def test_cpc_prediction_network_matches_manual_comp(embedding):
    torch.manual_seed(2)
    N, T, Cl, Cc, K, M = 5, 7, 9, 11, 3, 1000
    lens = torch.randint(K + 1, T + 1, (N,))
    lens[0] = T
    sum_lens = lens.sum().item()
    latent, context = torch.randn(sum_lens, Cl), torch.randn(N, T, Cc)
    samp = torch.randint(sum_lens, (N * (T - 1) * M,))
    latent_samp = latent[samp].view(N, T - 1, M, Cl)
    sources = torch.randint(N, (N,))

    net = CPCLossNetwork(Cl, Cc, K, M, N if embedding else None)
    net.eval()

    loss_exp, latents_ = 0.0, []
    norm = 0
    for n in range(N):
        lens_n = lens[n]
        latent_n, latent = latent[:lens_n], latent[lens_n:]
        latents_.append(latent_n)
        assert latent_n.size(0) == lens_n
        context_n = context[n, :lens_n]
        if embedding:
            context_n = context_n + net.embed(sources[n : n + 1])
        latent_samp_n = latent_samp[n]
        Az_n = net.ff(context_n)
        assert Az_n.shape == (lens_n, Cl * K)
        Az_n = Az_n.view(lens_n, K, Cl).transpose(0, 1)
        for k in range(1, K + 1):
            Az_n_k = Az_n[k - 1]  # (lens_n - 1, Cl)
            for t in range(lens_n - k - 1):
                Az_n_k_t = Az_n_k[t]  # (Cl,)
                latent_n_tpk = latent_n[t + k]  # (Cl,)
                num = Az_n_k_t @ latent_n_tpk
                assert num.numel() == 1
                latent_samp_n_t = latent_samp_n[t]  # (M, Cl)
                denom = (latent_samp_n_t @ Az_n_k_t).logsumexp(0)
                assert denom.numel() == 1
                loss_exp = loss_exp - (num - denom)
                norm += 1
    loss_exp = loss_exp / norm

    latents = torch.nn.utils.rnn.pad_sequence(latents_, True, -1)
    loss_act = net(latents, context, lens, sources)
    assert torch.isclose(loss_exp, loss_act, atol=1e-1)

