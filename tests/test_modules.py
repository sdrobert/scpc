from typing import Type
import pytest
import torch

from scpc.modules import *


@pytest.mark.parametrize(
    "Encoder",
    [
        CausalSelfAttentionEncoder,
        ConvEncoder,
        FeedForwardEncoder,
        IdentityEncoder,
        RecurrentEncoder,
        SelfAttentionEncoder,
    ],
    ids=[
        "CausalSelfAttentionEncoder",
        "ConvEncoder",
        "FeedForwardEncoder",
        "IdentityEncoder",
        "RecurrentEncoder",
        "SelfAttentionEncoder",
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


def test_causal_self_attention_encoder_is_causal():
    torch.manual_seed(1)
    N, T, H = 3, 13, 17
    x = torch.rand(N, T, H)
    encoder = CausalSelfAttentionEncoder(H, 2 * H, T // 2 - 1, num_heads=1)
    encoder.eval()
    x1 = encoder(x)[0][:, : T // 2]
    x2 = encoder(x[:, : T // 2])[0]
    assert x1.shape == x2.shape
    assert torch.allclose(x1, x2, atol=1e-5)
    x1 = x1[:, -2:]
    x2 = encoder(x[:, 1 : T // 2])[0][:, -2:]
    assert not torch.allclose(x1[:, 0], x2[:, 0], atol=1e-5)
    assert torch.allclose(x1[:, 1], x2[:, 1], atol=1e-5)


@pytest.mark.parametrize(
    "embedding", [True, False], ids=["w/-embedding", "w/o-embedding"]
)
@pytest.mark.parametrize("penc", ["ff", "recur", "csa"])
def test_cpc_prediction_network_matches_manual_comp(embedding, penc):
    torch.manual_seed(2)
    N, T, Cl, Cc, K, M = 5, 7, 9, 11, 3, 100
    lens = torch.randint(K + 1, T + 1, (N,))
    lens[0] = T
    sum_lens = lens.sum().item()
    latent, context = torch.randn(sum_lens, Cl), torch.randn(N, T, Cc)
    samp = torch.randint(sum_lens, (N * (T - 1) * M,))
    latent_samp = latent[samp].view(N, T - 1, M, Cl)
    speakers = torch.randint(N, (N,))

    if penc == "ff":
        penc = None
    elif penc == "csa":
        penc = CausalSelfAttentionEncoder(Cc, Cl * K, num_heads=1)
    else:
        penc = RecurrentEncoder(Cc, Cl * K)

    net = CPCLossNetwork(Cl, Cc, K, M, penc, N if embedding else None)
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
            context_n = context_n + net.embed(speakers[n : n + 1])
        latent_samp_n = latent_samp[n]
        Az_n = net.prediction_encoder(context_n.unsqueeze(0))[0].squeeze(0)
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
    loss_exp = loss_exp / norm  # + math.log(M + 1)

    latents = torch.nn.utils.rnn.pad_sequence(latents_, True, -1)
    loss_act = net(latents, context, lens, speakers)
    assert torch.isclose(loss_exp, loss_act, rtol=1e-1)
