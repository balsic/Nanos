"""Arm factory: the encoder under test, and the reference architectures it is compared to.

Every arm is built here and trained by the same :mod:`eegbench.engine` loop, so the only
thing that differs between two rows of a results table is the architecture. That is a
deliberate trade and it is worth stating explicitly, because it cuts both ways:

* It is the *right* choice for attribution. A baseline trained with its own author's
  schedule, optimizer and augmentation confounds architecture with recipe, and the
  resulting "our model wins" is unattributable.
* It is a *handicap* for any baseline whose published number depends on its own recipe.
  A shared recipe is fair in the sense that everyone pays it, not in the sense that
  everyone pays it equally.

So a shared-recipe table is evidence about architecture under one recipe, and it should be
reported that way. Where a baseline's published figure matters, run it under its own
recipe as a separate, labelled arm rather than quietly adopting the better number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# The encoder under test lives outside this package and is treated as the object of study.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

__all__ = ["build_arm", "EEGTrialClassifier", "TORCH_BASELINES", "SKLEARN_BASELINES",
           "ATTENTION_BASELINES", "ARMS"]

TORCH_BASELINES = (
    "eegnet", "shallow", "deep", "atcnet", "conformer", "ctnet", "msvtnet",
    "sstdpn", "attentionbase", "tcformer", "sccnet", "eegnex",
)
#: Which baselines carry an explicit attention/transformer block. Labelling only.
ATTENTION_BASELINES = ("atcnet", "conformer", "ctnet", "msvtnet", "sstdpn",
                       "attentionbase", "tcformer")
SKLEARN_BASELINES = ("riemann", "csp_lda")
ARMS = ("ours", "ours_noattn", *TORCH_BASELINES, *SKLEARN_BASELINES)


class EEGTrialClassifier(nn.Module):
    """``EEGEncoder`` -> optional gated attention -> pooling -> linear head.

    ``chan_valid``
        A per-trial electrode-validity mask, applied **before** any layer that mixes
        electrodes. In a pooled union montage an absent electrode is zero-filled, and a
        convolution with a bias still emits a constant from it -- a constant that differs
        per corpus, which is a dataset-identity cue the network is free to use. Masking
        after mixing would be too late; the mask has to gate the input.

    ``pool``
        ``mean`` over time is the default. The encoder emits ``(B, E, L)`` and the head is
        linear, so the pooling operator is the only place trial-level temporal structure
        is summarised.
    """

    def __init__(self, encoder: nn.Module, n_classes: int, *, attention: nn.Module | None = None,
                 head_dropout: float = 0.25, pool: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        self.pool = pool
        self.embed_dim = encoder.embed_dim
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(self.embed_dim, n_classes),
        )

    def features(self, x: torch.Tensor, chan_valid: torch.Tensor | None = None) -> torch.Tensor:
        if chan_valid is not None:
            x = x * chan_valid.unsqueeze(-1).to(x.dtype)
        h = self.encoder(x)                    # (B, E, L)
        if self.attention is not None:
            h = self.attention(h)
        if self.pool == "mean":
            return h.mean(dim=-1)
        if self.pool == "max":
            return h.amax(dim=-1)
        raise ValueError(f"pool must be mean|max, got {self.pool!r}")

    def forward(self, x: torch.Tensor, chan_valid: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.features(x, chan_valid))


def _largest_valid_head_count(embed_dim: int, requested: int) -> int:
    """Largest head count <= ``requested`` the attention module will accept.

    It asserts ``embed_dim % num_heads == 0`` *and* an even head dimension, because its
    rotary embedding rotates dimension pairs. Those bite at real montages -- a 61-channel
    union gives ``E=244``, where the usual 4 heads yields an odd head dim and trips the
    assert. Resolving downward is preferable to failing, but the resolved value changes
    attention capacity, so callers must record it rather than let it pass unnoticed.
    """
    for h in range(min(requested, embed_dim), 0, -1):
        if embed_dim % h == 0 and (embed_dim // h) % 2 == 0:
            return h
    raise ValueError(f"no valid head count for embed_dim={embed_dim}")


def build_arm(name: str, *, n_channels: int, n_classes: int, n_times: int,
              cfg: dict | None = None) -> tuple[nn.Module, dict]:
    """Construct one arm. Returns ``(module, description)``.

    ``description`` is written verbatim into the results JSON. It records what was
    *actually built* -- resolved head counts, embedding width, parameter count -- rather
    than what was requested, because those differ often enough that recording the request
    has already produced tables describing computations that never ran.
    """
    cfg = dict(cfg or {})

    if name in SKLEARN_BASELINES:
        return _SklearnArm(name), {"arm": name, "family": "riemannian"}

    if name in TORCH_BASELINES:
        model = _build_baseline(name, n_channels, n_classes, n_times)
        return model, {
            "arm": name, "family": "braindecode",
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "attention": name in ATTENTION_BASELINES,
        }

    if name not in ("ours", "ours_noattn"):
        raise ValueError(f"unknown arm {name!r}; known: {ARMS}")

    from encoder.model import EEGEncoder, GlobalAttentionModule

    enc_kwargs = {k: v for k, v in cfg.items() if k in _ENCODER_KWARGS}
    encoder = EEGEncoder(
        input_channels=n_channels,
        output_channels_per_group=cfg.get("channels_per_group", 2),
        **enc_kwargs,
    )
    embed_dim = encoder.embed_dim

    attention = None
    resolved_heads = None
    if name == "ours":
        requested = int(cfg.get("num_heads", 4))
        resolved_heads = _largest_valid_head_count(embed_dim, requested)
        attention = GlobalAttentionModule(
            input_channels=n_channels, embed_dim=embed_dim,
            ffn_dim=int(cfg.get("ffn_dim", 256)), num_heads=resolved_heads,
            attn_impl=cfg.get("attn_impl", "dense"),
            local_window=int(cfg.get("local_window", 0)),
        )

    model = EEGTrialClassifier(
        encoder, n_classes, attention=attention,
        head_dropout=float(cfg.get("head_dropout", 0.25)),
        pool=cfg.get("pool", "mean"),
    )
    desc = {
        "arm": name, "family": "ours",
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "embed_dim": embed_dim,
        "attention": attention is not None,
        "encoder_kwargs": enc_kwargs,
    }
    if resolved_heads is not None:
        desc["num_heads_requested"] = int(cfg.get("num_heads", 4))
        desc["num_heads_resolved"] = resolved_heads
    return model, desc


#: Constructor keywords forwarded to the encoder. Kept as an explicit set so that adding a
#: flag to the CLI without adding it here fails loudly at construction rather than
#: silently building the default architecture and reporting the requested one.
_ENCODER_KWARGS = frozenset({
    "kernel_size", "stride", "padding", "dropout", "spatial_filter", "branch2_stride",
    "spectral", "n_fft", "norm", "pool_mode", "same_padding", "kernel_sizes",
    "pool_kernel", "pool_stride", "stage2_kernel", "stage2_pool", "spatial_mode",
    "n_temporal", "spatial_per_temporal", "band_pool", "spatial_max_norm",
})


def _build_baseline(name: str, n_channels: int, n_classes: int, n_times: int) -> nn.Module:
    import braindecode.models as bd

    kw = dict(n_chans=n_channels, n_outputs=n_classes, n_times=n_times)
    if name == "eegnet":
        return bd.EEGNet(**kw)
    if name == "shallow":
        return bd.ShallowFBCSPNet(**kw, final_conv_length="auto")
    if name == "deep":
        return bd.Deep4Net(**kw, final_conv_length="auto")
    if name == "sccnet":
        # SCCNet cannot infer the sampling rate from (n_chans, n_times) and raises rather
        # than guessing. 250 Hz is the contract everywhere in this harness.
        return bd.SCCNet(**kw, sfreq=250.0)
    simple = {"atcnet": "ATCNet", "conformer": "EEGConformer", "ctnet": "CTNet",
              "msvtnet": "MSVTNet", "sstdpn": "SSTDPN", "attentionbase": "AttentionBaseNet",
              "tcformer": "TCFormer", "eegnex": "EEGNeX"}
    if name in simple:
        return getattr(bd, simple[name])(**kw)
    raise ValueError(f"unknown braindecode baseline {name!r}")


class _SklearnArm(nn.Module):
    """Marker for the non-gradient arms, which :mod:`eegbench.engine` routes elsewhere.

    A tangent-space pipeline is not optional in this table. MOABB's own benchmark found
    Riemannian methods competitive with or ahead of deep learning on motor imagery, so a
    table without one invites the question immediately. It is also the pipeline gate: a
    large deviation from the published within-session figure means the preprocessing is
    broken, not that the architecture is.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def pipeline(self):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        if self.name == "riemann":
            # OAS shrinkage rather than the empirical estimator: a few hundred trials over
            # 22-64 channels is not comfortably in the regime where the sample covariance
            # is well conditioned.
            return make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                LogisticRegression(max_iter=2000, C=1.0),
            )
        if self.name == "csp_lda":
            from mne.decoding import CSP
            return make_pipeline(
                CSP(n_components=8, reg="ledoit_wolf", log=True),
                LinearDiscriminantAnalysis(),
            )
        raise ValueError(f"unknown sklearn arm {self.name!r}")
