import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

_FLEX_COMPILED: dict = {}


def _compiled_flex(name):
    """Lazily ``torch.compile`` a flex_attention entry point, once per process.

    Compilation is what turns these from score-materializing reference implementations
    into fused kernels; see ``GlobalAttentionModule._attend_flex``. Cached because
    recompiling per module instance would pay the (substantial) Dynamo cost on every
    construction.
    """
    if name not in _FLEX_COMPILED:
        import torch.nn.attention.flex_attention as fa
        _FLEX_COMPILED[name] = torch.compile(getattr(fa, name), dynamic=False)
    return _FLEX_COMPILED[name]


class AlignmentPrefilter(nn.Module):
    """In-architecture covariance alignment: ``x -> R^(-1/2) x``.

    Euclidean / Riemannian Alignment is normally applied as a preprocessing step outside
    the model. Folding it in changes nothing numerically -- it is the same linear map -- but
    it makes the model **self-contained**: the whitener travels in the checkpoint, so a
    saved model carries its own alignment and inference is a single forward call rather
    than a pipeline someone has to remember to reproduce. That matters for deployment on a
    new subject, which is the setting this whole line of work is aimed at.

    Two ways to obtain the whitener:

    * ``fit(x)`` -- estimate it from a batch of that subject's trials. Uses **no labels**,
      so it is legitimate to run on a new subject's raw recording.
    * ``load_state_dict`` -- reuse one fitted earlier for the same subject.

    The whitener is a registered buffer, not a parameter: it is a statistic of the input
    distribution, not something gradient descent should be moving. Making it a parameter
    would let the network quietly un-do the alignment during training.

    ``mode='riemann'`` re-centres on the Riemannian (Frechet) mean of the trial covariances
    rather than the arithmetic mean. Covariances live on the SPD manifold, where the
    arithmetic mean is only a first-order approximation to the centre of mass and is biased
    toward high-variance trials.
    """

    def __init__(self, n_channels: int, mode: str = "euclid", rank_tol: float = 1e-10):
        super().__init__()
        if mode not in ("none", "euclid", "riemann"):
            raise ValueError(f"mode must be none|euclid|riemann, got {mode!r}")
        self.mode = mode
        self.rank_tol = rank_tol
        self.register_buffer("whitener", torch.eye(n_channels))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "AlignmentPrefilter":
        """Estimate the whitener from ``(N, C, T)`` trials. Label-free."""
        if self.mode == "none":
            return self
        from eegbench.align import fit_whitener

        # The whitener is taken directly from the fitting routine rather than recovered
        # from aligned data by least squares. The recovery form was well-posed only when
        # the covariance was full rank: on a union montage, where absent electrodes make
        # it genuinely singular, `lstsq` returns *a* solution to an underdetermined system
        # rather than *the* whitener, and the two differ exactly in the null space -- the
        # zero-filled channels. That produced a plausible matrix and a silently wrong map.
        arr = x.detach().float().cpu().numpy()
        w = fit_whitener(arr, mode=self.mode, rank_tol=self.rank_tol)
        self.whitener.copy_(
            torch.from_numpy(w.matrix.astype("float32")).to(self.whitener.device)
        )
        self.fitted.fill_(True)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        if not bool(self.fitted):
            # Without this the module applies torch.eye -- a silent no-op -- while
            # `describe()` writes "align": "euclid" into the results JSON. That is the
            # fourth instance of this failure class in the project (--select best-val on an
            # empty validation set, the empty session-name selection, --norm euclid on a
            # sklearn arm), and all three of the others produced result files describing a
            # computation that never happened.
            raise RuntimeError(
                f"AlignmentPrefilter(mode={self.mode!r}) was never fitted: the whitener is "
                "still the identity. Call .fit(trials) on the subject's own unlabelled "
                "data, or load a state_dict that carries a fitted whitener."
            )
        return torch.einsum("cd,ndt->nct", self.whitener.to(x.dtype), x)


class LogVarPool(nn.Module):
    """``square -> AvgPool -> log``: ShallowFBCSPNet's feature, as a drop-in for GELU+MaxPool.

    The trunk as published computes ``mean_t(MaxPool_k(GELU(conv)))``. That is monotone in
    band power, so it is not true that it "cannot represent" it -- the problem is
    conditioning. Measured on the trained champion (``eeg_bench.diagnose pooling``): the
    pooled features have kurtosis 6.5 against a Gaussian's 3, and a +/-20% per-channel gain
    moves the feature the head actually sees by **74%** of its own magnitude.

    Averaging the *square* and taking the log fixes the second directly. A per-channel gain
    ``a`` maps a power feature ``f -> a^2 f``, which the log turns into ``f + 2 log a`` -- an
    additive offset any per-feature centering absorbs, instead of a multiplicative one no
    shared affine head can. Measured on the same weights, swapping only this operator:
    gain sensitivity 0.735 -> 0.110 (**6.7x**), kurtosis 6.46 -> 5.05, and with EA the
    features become very nearly Gaussian (kurtosis 3.53, skew 0.39).

    What it does **not** do, measured rather than assumed: it leaves the between-subject
    over within-subject feature-variance ratio unchanged (0.734 vs 0.715). Per-channel
    *gain* is evidently not the dominant form of inter-subject variability -- covariance
    structure is, which is what Euclidean Alignment corrects. Claim this operator for
    conditioning and gain robustness, not for cross-subject transfer.

    ``eps`` floors the average power before the log. It is additive rather than a clamp
    because absent electrodes in a pooled union montage are exactly zero-power, and
    ``log(0)`` would propagate ``-inf`` through the whole embedding.
    """

    def __init__(self, kernel_size, stride=1, eps=1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps

    def forward(self, x):
        return torch.log(F.avg_pool1d(x.pow(2), self.kernel_size, self.stride) + self.eps)

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, stride={self.stride}, eps={self.eps}"


class MaxNorm(nn.Module):
    """Rescale rows whose L2 norm exceeds ``max_val``. EEGNet's spatial-filter constraint.

    Applied through ``torch.nn.utils.parametrize``, so the constraint is enforced on every
    forward rather than as a post-step projection that an optimizer state can drift out of.

    The reason this is on the list at all: the champion mixes 22 electrodes into 88 filters
    with no constraint and no bias -- a 4x over-complete basis. FINDINGS_v2 section 9d
    justified the width by analogy to CSP and then measured it buying +0.37 (n.s.) on test
    for 4x the parameters. An unconstrained over-complete spatial basis fitted on one
    subject is precisely the component that encodes subject-specific covariance structure,
    which is the thing that fails to transfer; EEGNet constrains its spatial conv for
    exactly this reason.
    """

    def __init__(self, max_val=1.0, axis=0, eps=1e-8):
        super().__init__()
        self.max_val = max_val
        self.axis = axis
        self.eps = eps

    def forward(self, w):
        dims = [d for d in range(w.dim()) if d != self.axis]
        norm = w.norm(2, dim=dims, keepdim=True).clamp_min(self.eps)
        return w * (norm.clamp(max=self.max_val) / norm)

    def extra_repr(self):
        return f"max_val={self.max_val}, axis={self.axis}"


def _norm_layer(kind: str, num_features: int) -> nn.Module | None:
    """``none`` | ``batch`` | ``instance``, or raise. Returns ``None`` for ``none``."""
    if kind == "none":
        return None
    if kind == "batch":
        return nn.BatchNorm1d(num_features)
    if kind == "instance":
        return nn.InstanceNorm1d(num_features, affine=True)
    raise ValueError(f"norm must be none|batch|instance, got {kind!r}")


class SpectralBranch(nn.Module):
    """Log-power STFT branch: ``(B, C, S) -> (B, opg*C, T')``.

    The two convolutional branches reach frequency only implicitly, through dilation.
    Neural oscillations are *defined* in the frequency domain, so handing the trunk an
    explicit spectral view costs one STFT and removes the depth otherwise needed to
    rediscover it. Kept per-channel (``groups=C``) so it composes with the same
    ``feature_channel_index`` contract the convolutional branches obey.

    Each channel's ``n_fft//2+1`` log-power bins are projected to ``opg`` features by a
    grouped 1x1 conv, so the branch adds ``opg*C`` dims -- the same width as each
    convolutional branch, making ``E = 3*opg*C`` when enabled.
    """

    def __init__(self, input_channels, output_channels_per_group, n_fft=64, hop_length=None):
        super().__init__()
        self.input_channels = input_channels
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.n_bins = n_fft // 2 + 1
        # Hann window registered as a buffer so .to(device) carries it.
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.project = nn.Conv1d(
            in_channels=self.n_bins * input_channels,
            out_channels=output_channels_per_group * input_channels,
            kernel_size=1,
            groups=input_channels,
        )

    def forward(self, x):
        B, C, S = x.shape
        # torch.stft wants (N, S); fold channels into the batch axis.
        flat = x.reshape(B * C, S)
        # center=True pads by n_fft//2 so short trials still yield frames; return_complex
        # is mandatory in torch>=2. Cast to float32: stft rejects half precision.
        spec = torch.stft(
            flat.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )  # (B*C, n_bins, T')
        # log1p of power rather than log: power has genuine zeros at masked/absent
        # electrodes, and log(0) would propagate -inf through the whole embedding.
        power = spec.real.pow(2) + spec.imag.pow(2)
        feats = torch.log1p(power)
        T = feats.shape[-1]
        feats = feats.reshape(B, C * self.n_bins, T).to(x.dtype)
        return self.project(feats)  # (B, opg*C, T')


class BandSpatialBranch(nn.Module):
    """Temporal filtering first, then a *separate* spatial filter per temporal filter.

    This is the ordering EEGNet, ATCNet and the FBCSP family all use, and the one the
    original trunk does not have. It matters for motor imagery specifically:

    * mu (~10 Hz) and beta (~20 Hz) desynchronisation have **different scalp topographies**;
    * the published trunk mixes electrodes once, on the raw broadband signal, and then
      processes every mixed channel in isolation (``groups=input_channels``), so a single
      spatial basis has to serve every rhythm at once;
    * here each of ``n_temporal`` temporal filters gets its own ``spatial_per_temporal``
      spatial filters (a depthwise conv over the electrode axis, ``groups=n_temporal``),
      so the spatial pattern can be *band-specific*.

    That is the concrete reason a 2,932-parameter EEGNet matches a 15,668-parameter
    version of this trunk: not capacity, but where the spatial stage sits.

    Shapes: ``(B, C, T) -> (B, n_temporal * spatial_per_temporal, T')``.
    """

    def __init__(self, n_channels, n_temporal, spatial_per_temporal, kernel_size,
                 dilation=1, dropout=0.25, pool=8, max_norm=1.0):
        super().__init__()
        self.max_norm = max_norm
        pad = (dilation * (kernel_size - 1)) // 2
        # Temporal: one filter bank shared across electrodes, so each output channel is a
        # frequency-selective view of the whole montage.
        self.temporal = nn.Conv2d(1, n_temporal, (1, kernel_size),
                                  dilation=(1, dilation), padding=(0, pad), bias=False)
        self.bn_t = nn.BatchNorm2d(n_temporal)
        # Spatial: depthwise over the electrode axis. groups=n_temporal is what makes the
        # spatial filters *per temporal filter* rather than shared.
        self.spatial = nn.Conv2d(n_temporal, n_temporal * spatial_per_temporal,
                                 (n_channels, 1), groups=n_temporal, bias=False)
        self.bn_s = nn.BatchNorm2d(n_temporal * spatial_per_temporal)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d((1, pool))
        self.drop = nn.Dropout(dropout)

    def _constrain(self):
        # Max-norm on the spatial kernels: the standard regulariser for depthwise spatial
        # convs on EEG, and what keeps a single electrode from dominating a filter.
        if self.max_norm and self.training:
            with torch.no_grad():
                w = self.spatial.weight
                # Per-output-filter L2 over its (1, C, 1) kernel. torch.norm rejects a
                # 3-tuple dim, so flatten everything except the filter axis first.
                n = w.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1, 1)
                w.mul_(n.clamp(max=self.max_norm) / n)

    def forward(self, x):
        self._constrain()
        h = x.unsqueeze(1)                 # (B, 1, C, T)
        h = self.bn_t(self.temporal(h))    # (B, F, C, T)
        h = self.act(self.bn_s(self.spatial(h)))  # (B, F*D, 1, T)
        h = self.drop(self.pool(h))
        return h.squeeze(2)                # (B, F*D, T')


class EEGEncoder(nn.Module):
    """EEG encoding front-end. Algorithm from https://arxiv.org/pdf/2509.20489.

    Every keyword defaults to the as-published behaviour, so an unmodified call
    reproduces the numbers already in ``results/eeg_bench/``. The additions exist because
    three deficits were *measured* on BCI IV 2a (see ``results/eeg_bench/FINDINGS.md``);
    each is opt-in so it can be ablated one at a time.

    ``dropout`` (measured: -10.6 pts at the hardcoded 0.5)
        Was ``Dropout1d(p=0.5)`` in three places with no way to change it.
        ``Dropout1d`` zeroes *whole feature channels*, so at 0.5 roughly half the
        per-electrode groups vanish, three times over.

    ``spatial_filter`` (measured: ~+8 pts)
        ``groups=input_channels`` means the convolutional front-end **never mixes
        electrodes** -- all cross-electrode integration is deferred to the single
        attention block. Motor imagery is decoded almost entirely from spatial patterns.
        A 1x1 conv before the trunk supplies the missing stage. NOTE: when this is
        active, trunk "channels" are learned mixtures, not electrodes, so
        ``eeg_bench.encoder.feature_channel_index`` no longer maps back to input
        electrodes and channel-validity masking must be applied *before* this layer.

    ``branch2_stride`` (structural defect, previously undiagnosed)
        As written, branch 2 runs ``stride=4*stride`` on both its conv and its maxpool.
        At ``kernel_size=32`` on a 750-sample trial that leaves **32 timepoints**, which
        ``adaptive_avg_pool1d`` then stretches 21.6x to 690 -- so 44 of the 88 embedding
        dims are piecewise-constant over ~22-sample runs. The source comment called the
        stretch a ``temp sol``. ``dilation=4`` is what buys branch 2 its longer
        timescale; the *stride* only discards resolution. Pass ``branch2_stride=1`` to
        keep the dilation and drop the decimation.

    ``spectral``
        Adds :class:`SpectralBranch`, changing ``E`` from ``2*opg*C`` to ``3*opg*C``.

    ``norm`` (never present in any published arm)
        The trunk as written contains **no normalization layer of any kind** --
        ``[Conv1d, Dropout1d, GELU, MaxPool1d]`` twice and nothing else. Every reference
        architecture it is benchmarked against has one: EEGNet is ``Conv -> BatchNorm ->
        ELU -> AvgPool``, ShallowFBCSPNet is ``Conv -> BatchNorm -> square -> AvgPool ->
        log``. This is the one element of the standard motor-imagery feature recipe the
        trunk has never had.

        Placement follows ShallowFBCSPNet exactly: immediately after the conv, *before* the
        pooling operator. ``batch`` is that architecture's own choice and is safe with
        ``pool_mode='logvar'`` because BatchNorm normalizes across the batch per feature and
        so leaves within-trial power intact. **``instance`` normalizes each trial over
        time**, which removes each feature's total power -- exactly the quantity a
        log-variance pool then measures. It is implemented for ablation and because it is
        the montage-agnostic option for pooled cross-dataset arms, but a null result for
        ``instance + logvar`` should be read as this interaction rather than as evidence
        about normalization.

    ``pool_mode``
        ``maxgelu`` is as published. ``logvar`` swaps in :class:`LogVarPool` over the same
        window, so the receptive field is unchanged and this is a like-for-like swap of the
        operator rather than a different model. See that class for what it was measured to
        buy and, more importantly, what it was measured not to.

    ``same_padding`` (defect, previously undiagnosed)
        ``padding=1`` is hardcoded regardless of ``kernel_size`` or ``dilation``, so at
        ``kernel_size=32, branch2_stride=1`` branch 1 emits **690** timepoints and branch 2
        emits **597**, and ``adaptive_avg_pool1d`` then resamples one onto the other. Index
        *t* in the two halves of ``E`` therefore refers to different absolute latencies --
        both an offset (the receptive fields start ~46 samples apart) and a scale mismatch.
        For a module whose entire job is attending over that axis, that is a real defect.
        ``same_padding=True`` pads each branch by its own ``dilation*(kernel-1)``, so both
        convs are centred and both branches emit the same length with the same latency, and
        the resample becomes a no-op. Capacity-neutral: no parameter is added.
    """

    def __init__(self, input_channels, output_channels_per_group, kernel_size=4, stride=1,
                 padding=1, dropout=0.5, spatial_filter=0, branch2_stride=None,
                 spectral=False, n_fft=64, norm="none", pool_mode="maxgelu",
                 same_padding=False, kernel_sizes=None, pool_kernel=None, pool_stride=1,
                 stage2_kernel=0, stage2_pool=1, spatial_mode="pre", n_temporal=8,
                 spatial_per_temporal=2, band_pool=8, spatial_max_norm=0.0):
        super(EEGEncoder, self).__init__()
        if pool_mode not in ("maxgelu", "logvar"):
            raise ValueError(f"pool_mode must be maxgelu|logvar, got {pool_mode!r}")

        # --- band-specific spatial filtering -------------------------------------------
        # `spatial_mode="per_filter"` reorders the front end: filter in TIME first, then
        # learn spatial filters *per temporal filter* (BandSpatialBranch). The published
        # trunk does the opposite -- one 1x1 electrode mix on the raw broadband signal,
        # then per-channel convs (groups=input_channels) that never mix again -- so one
        # spatial basis has to serve mu and beta at once, though their topographies differ.
        # The two dilations (1 and 4) are kept, so the multi-scale identity survives.
        self.spatial_mode = spatial_mode
        if spatial_mode == "per_filter":
            # This path builds BandSpatialBranch, which hardcodes its own BatchNorm2d,
            # ELU and AvgPool. It therefore IGNORES the trunk's norm / pool_mode /
            # same_padding / multi-scale / stage-2 settings -- but the caller records the
            # requested kwargs verbatim into the results JSON, so an ignored `norm="batch"`
            # is written into a file describing a model that contains no BatchNorm1d at
            # all. Refuse rather than mis-record: this is the project's characteristic
            # failure, a wrong number rather than a crash.
            _ignored = {
                "norm": (norm, "none"), "pool_mode": (pool_mode, "maxgelu"),
                "same_padding": (same_padding, False), "kernel_sizes": (kernel_sizes, None),
                "pool_kernel": (pool_kernel, None), "pool_stride": (pool_stride, 1),
                "stage2_kernel": (stage2_kernel, 0), "stage2_pool": (stage2_pool, 1),
                "spatial_filter": (spatial_filter, 0),
            }
            _bad = {k: v for k, (v, default) in _ignored.items() if v != default}
            if _bad:
                raise ValueError(
                    f"spatial_mode='per_filter' ignores {sorted(_bad)} (it builds "
                    "BandSpatialBranch, which has its own normalization and pooling), but "
                    f"they were set to {_bad}. They would be recorded in the results file "
                    "as if applied. Drop them, or use spatial_mode='pre'."
                )
            self.spatial = None
            self.band_one = BandSpatialBranch(
                input_channels, n_temporal, spatial_per_temporal, kernel_size,
                dilation=1, dropout=dropout, pool=band_pool)
            self.band_two = BandSpatialBranch(
                input_channels, n_temporal, spatial_per_temporal, kernel_size,
                dilation=4, dropout=dropout, pool=band_pool)
            self.spectral = (
                SpectralBranch(input_channels, output_channels_per_group, n_fft=n_fft)
                if spectral else None)
            self.n_branches = 3 if spectral else 2
            self.embed_dim = 2 * n_temporal * spatial_per_temporal + (
                output_channels_per_group * input_channels if spectral else 0)
            self.input_channels = input_channels
            self.trunk_channels = input_channels
            self.output_channels_per_group = output_channels_per_group
            self.kernel_size = kernel_size
            self.branch2_stride = None
            self.dropout = nn.Dropout1d(p=dropout, inplace=False)
            # The early return skips the shared setup below, so every attribute the
            # classifier's describe() reads off the encoder has to be set here too --
            # otherwise this mode builds and trains fine and only fails at reporting time.
            self.pool_kernel = pool_kernel
            self.pool_stride = pool_stride
            self.stage2_kernel = stage2_kernel
            self.stage2_pool = stage2_pool
            self.norm = norm
            self.pool_mode = pool_mode
            self.same_padding = same_padding
            self.kernel_sizes = tuple(kernel_sizes) if kernel_sizes else None
            self.n_temporal = n_temporal
            self.spatial_per_temporal = spatial_per_temporal
            self.band_pool = band_pool
            return
        # Multi-scale mode replaces the two-branch (dilation 1, dilation 4) structure with
        # one branch per kernel size, all undilated -- MSVTNet's front end, which is the
        # arm that beats us on both benchmarks (+2.91 on Cho2017+EA, 10/10 folds,
        # p=0.0020). Its kernels are (15, 31, 63, 125) at 250 Hz: 60 ms to 500 ms. Ours
        # reaches 500 ms too (kernel 32 at dilation 4 spans 125 samples) but has nothing
        # SHORT -- both branches share one kernel, so the scale range is spanned by
        # dilation alone and the fine end is missing.
        #
        # Every branch shares one pooling window, so they all emit the same length and
        # concatenate without the resample the two-branch path needs.
        self.kernel_sizes = tuple(kernel_sizes) if kernel_sizes else None
        if self.kernel_sizes:
            if len(self.kernel_sizes) < 1:
                raise ValueError("kernel_sizes must name at least one branch")
            if not same_padding:
                raise ValueError(
                    "kernel_sizes requires same_padding=True: without it each branch emits "
                    "a different length and one would be resampled onto another, which is "
                    "the defect this path exists to avoid"
                )
        self.pool_kernel = pool_kernel if pool_kernel is not None else kernel_size
        if same_padding and not kernel_sizes and (
            stride * 4 if branch2_stride is None else branch2_stride) != 1:
            # The whole point of same_padding is that both branches emit one length at one
            # latency. Branch 2 still decimates by branch2_stride, so at the as-published
            # 4x it emits 40 timepoints against branch 1's 719 -- an 18x stretch, worse
            # than the 690/597 mismatch the flag exists to remove. Refuse rather than
            # record same_padding: true for a configuration where it does not hold.
            raise ValueError(
                "same_padding equalizes branch lengths only when branch2_stride=1; with "
                f"branch2_stride={stride * 4 if branch2_stride is None else branch2_stride} "
                "branch 2 still decimates and one branch is resampled onto the other. "
                "Pass branch2_stride=1, or drop same_padding."
            )
        # The pool slides by 1, so the trunk emits 690 positions for a 750-sample trial and
        # neighbours share 31 of 32 samples -- measured at 0.987 cosine
        # (`eeg_bench.diagnose attention`). MSVTNet, which beats us on both benchmarks,
        # decimates 56x and hands its transformer 18 tokens. `pool_stride` decouples the
        # pooling *window* (which sets the receptive field) from the *hop* (which sets how
        # much of the resulting redundancy is kept), so the two can be varied separately
        # instead of both being pinned by `stride`. 1 is the as-published hop.
        self.pool_stride = pool_stride
        # A SECOND depthwise temporal stage, after the first pool. The trunk is depth-1 per
        # branch; every architecture that beats it is deeper. MSVTNet's TSConv block is
        # conv -> BN -> spatial conv -> BN -> ELU -> pool(8) -> conv(15) -> BN -> ELU ->
        # pool(7), i.e. two temporal stages with decimation between them, and it is +2.91
        # over us on Cho2017+EA (10/10 folds, p=0.0020).
        #
        # `groups=out_channels` is load-bearing, not a default: a dense conv here would mix
        # features across electrodes and silently break the contract
        # `eeg_bench.encoder.feature_channel_index` states -- every output feature depends on
        # exactly one input channel -- which the channel-validity masking for pooled union
        # montages relies on. Depthwise keeps that map exact.
        self.stage2_kernel = stage2_kernel
        self.stage2_pool = stage2_pool

        # 1x1 electrode mixing, ahead of the grouped convolutions that cannot do it.
        self.spatial_filter = spatial_filter
        self.spatial_max_norm = float(spatial_max_norm)
        if spatial_filter:
            self.spatial = nn.Conv1d(input_channels, spatial_filter, kernel_size=1, bias=False)
            if self.spatial_max_norm > 0:
                # Constrain each spatial filter's row to unit L2, EEGNet's regulariser on
                # its own spatial stage. The reason it belongs here specifically: this
                # layer is a C -> F dense mix with F typically well above C -- a 4x
                # over-complete basis on a 22-electrode montage -- with no constraint and
                # no bias. An over-complete spatial basis fitted freely is the component
                # most able to encode a particular head's anatomy rather than the task,
                # and the measured signature matches: this trunk beats EEGNet
                # within-subject and loses to it across subjects.
                #
                # Applied through `parametrize` rather than as a post-step projection, so
                # the constraint holds on every forward instead of only at points where
                # an optimizer happens to have been stepped.
                from torch.nn.utils import parametrize
                parametrize.register_parametrization(
                    self.spatial, "weight", MaxNorm(self.spatial_max_norm, axis=0))
            trunk_channels = spatial_filter
        else:
            self.spatial = None
            trunk_channels = input_channels

        self.input_channels = input_channels
        self.trunk_channels = trunk_channels
        self.output_channels_per_group = output_channels_per_group
        self.kernel_size = kernel_size
        # None preserves the as-written 4x decimation; 1 keeps full temporal resolution.
        self.branch2_stride = stride * 4 if branch2_stride is None else branch2_stride

        self.norm = norm
        self.pool_mode = pool_mode
        self.same_padding = same_padding
        out_channels = output_channels_per_group * trunk_channels

        def branch(dilation, br_stride, k=kernel_size, pool_k=None):
            pool_k = kernel_size if pool_k is None else pool_k
            # br_stride is the conv's decimation (branch2's 4x, historically applied to the
            # pool as well). The pool's own hop multiplies it, so pool_stride=1 reproduces
            # the published behaviour exactly for both branches.
            pool_hop = br_stride * self.pool_stride
            layers: list[nn.Module] = []
            conv_padding = padding
            if same_padding:
                # An even kernel cannot be centred symmetrically, so the extra sample goes
                # on the right. Done with an explicit pad rather than Conv1d(padding=...)
                # because that argument only takes a symmetric amount.
                total = dilation * (k - 1)
                layers.append(nn.ConstantPad1d((total // 2, total - total // 2), 0.0))
                conv_padding = 0
            layers.append(nn.Conv1d(
                in_channels=trunk_channels,
                # likely since needs to be per channel basis for each encoder
                out_channels=out_channels,
                kernel_size=k,
                dilation=dilation,
                stride=br_stride,
                padding=conv_padding,
                groups=trunk_channels,
            ))
            norm_layer = _norm_layer(norm, out_channels)
            if norm_layer is not None:
                layers.append(norm_layer)
            def add_pool(k, hop):
                if pool_mode == "logvar":
                    # Dropout AFTER the log, not before: Dropout1d zeroes whole feature
                    # channels, and a zero entering `log` is the -inf that `eps` exists to
                    # prevent. After the log a zeroed channel reads as unit power, which is
                    # a sane neutral value.
                    layers.append(LogVarPool(k, hop))
                    layers.append(nn.Dropout1d(p=dropout, inplace=True))
                else:
                    layers.append(nn.Dropout1d(p=dropout, inplace=True))
                    layers.append(nn.GELU())
                    layers.append(nn.MaxPool1d(kernel_size=k, stride=hop))

            add_pool(pool_k, pool_hop)
            if self.stage2_kernel:
                # Same-pad so the stage cannot silently shorten the sequence, and depthwise
                # over the feature axis so the feature->electrode map survives.
                s2 = self.stage2_kernel
                layers.append(nn.ConstantPad1d(((s2 - 1) // 2, (s2 - 1) - (s2 - 1) // 2), 0.0))
                layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=s2,
                                        groups=out_channels))
                n2 = _norm_layer(norm, out_channels)
                if n2 is not None:
                    layers.append(n2)
                add_pool(self.stage2_pool, self.stage2_pool)
            return nn.Sequential(*layers)

        if self.kernel_sizes:
            if len(self.kernel_sizes) < 2:
                raise ValueError("kernel_sizes needs at least 2 scales to be multi-scale")
            # One branch per scale, all undilated, all sharing self.pool_kernel so the
            # outputs concatenate without a resample.
            #
            # The first two keep the names `encoder_one` / `encoder_two`, and scales beyond
            # them go in a separate ModuleList. Two reasons, both about not corrupting
            # things silently:
            #   * every saved checkpoint has `encoder_one.*` / `encoder_two.*` keys, so
            #     renaming the storage would break strict loading on all 45 of them;
            #   * binding the same module to BOTH `encoder_one` and `branches[0]` registers
            #     it twice, which emits duplicate state_dict keys -- measured at 12 keys for
            #     8 tensors, so a parameter count taken over the state_dict over-reports by
            #     20%. `parameters()` deduplicates and the optimizer was fine, but a wrong
            #     number in a table is exactly the failure this repo keeps catching.
            mods = [branch(1, stride, k=k, pool_k=self.pool_kernel) for k in self.kernel_sizes]
            self.encoder_one, self.encoder_two = mods[0], mods[1]
            self.extra_branches = nn.ModuleList(mods[2:])
        else:
            self.encoder_one = branch(1, stride, pool_k=self.pool_kernel)
            self.encoder_two = branch(4, self.branch2_stride, pool_k=self.pool_kernel)
            self.extra_branches = nn.ModuleList()

        self.spectral = (
            SpectralBranch(trunk_channels, output_channels_per_group, n_fft=n_fft)
            if spectral else None
        )
        n_conv_branches = len(self.kernel_sizes) if self.kernel_sizes else 2
        self.n_branches = n_conv_branches + (1 if spectral else 0)
        self.embed_dim = self.n_branches * output_channels_per_group * trunk_channels

        self.dropout = nn.Dropout1d(p=dropout, inplace=True)

    def forward(self, x: torch.tensor):
        if x.dim()==2:
            x = x.unsqueeze(0)

        if self.spatial_mode == "per_filter":
            b1, b2 = self.band_one(x), self.band_two(x)
            # dilation 4 emits a shorter sequence; match branch 1 as the other paths do.
            b2 = F.adaptive_avg_pool1d(b2, output_size=b1.shape[-1])
            outs = [b1, b2]
            if self.spectral is not None:
                outs.append(F.adaptive_avg_pool1d(self.spectral(x),
                                                  output_size=b1.shape[-1]))
            return self.dropout(torch.concat(outs, dim=1))

        if self.spatial is not None:
            x = self.spatial(x)

        if self.kernel_sizes:
            # Multi-scale: every branch already shares one pooling window and one padding
            # rule, so the lengths agree by construction and nothing is resampled.
            branches = [self.encoder_one(x), self.encoder_two(x)]
            branches += [b(x) for b in self.extra_branches]
        else:
            encoder_one_out: torch.tensor = self.encoder_one(x)
            encoder_two_out: torch.tensor = self.encoder_two(x)

            # include separate banches instead of pooling layers together, temp sol
            encoder_two_out = F.adaptive_avg_pool1d(encoder_two_out, output_size=encoder_one_out.shape[-1])
            # print(f"AFTER RESAHPE\nshape: {encoder_one_out.shape}, shape2: {encoder_two_out.shape}]\n")

            branches = [encoder_one_out, encoder_two_out]
        if self.spectral is not None:
            spec_out = self.spectral(x)
            branches.append(
                F.adaptive_avg_pool1d(spec_out, output_size=branches[0].shape[-1])
            )

        output = torch.concat(branches, dim=1)
        output = self.dropout(output)
        return output

class GlobalAttentionModule(nn.Module):
    """
    Global Gated Attention (Algorithm 2 / Section 3.2 / Fig. 1B of CoSupFormer,
    arXiv:2509.20489), implemented as a drop-in module that consumes the
    ``EEGEncoder`` output.

    -----------------------------------------------------------------------
    Notation mapping (paper -> this codebase)
    -----------------------------------------------------------------------
    The paper feeds attention a tensor ``z in R^{(CL) x E}``: ``CL`` "tokens"
    each carrying an ``E``-dim embedding, producing a ``CL x CL`` attention
    matrix. ``EEGEncoder.forward`` here emits ``(B, 4C, L1)``
    (batch, feature-channels, time). We map:

        * token / sequence axis  = ``L1`` (time)   -> attention mixes over this
        * embedding dim ``E``    = ``4C`` (feature-channels, e.g. 240 for C=60)
        * head dim ``d``         = ``E / num_heads``
        * attention matrix       = ``L1 x L1`` per head

    So ``forward`` transposes ``(B, 4C, L1) -> (B, L1, E=4C)``, runs Algorithm 2,
    then transposes back to ``(B, 4C, L1)`` so ``output.shape == input.shape``.
    Both batched ``(B, 4C, L1)`` and unbatched ``(4C, L1)`` inputs are accepted.

    -----------------------------------------------------------------------
    Algorithm 2 line -> code-section mapping
    -----------------------------------------------------------------------
        line 1  z_bar = LayerNorm(z)              -> self.layer_norm (over E=4C)
        line 2  q,k,v = LinearNoBias(z_bar)       -> q/k/v_proj_linear (bias-free)
        line 3  q = RoPE(q); k = RoPE(k)          -> _apply_rope (per head, over L1)
        line 4  g = sigmoid(Linear(z_bar))        -> self.gate_linear (WITH bias)
        line 5  a = softmax_i(mask((1/sqrt(d)) q^T k))
                                                  -> scaled dot-product + diagonal
                                                     -inf mask + softmax over keys
        line 6  z = LayerNorm(a @ v + z_bar)      -> self.norm_attn
        line 7  z = LayerNorm(z + FFN(z))         -> self.fnn + self.norm_ffn
        line 8  o = Linear(z)                     -> self.out_proj_linear (WITH bias)
        line 9  o = o * g                         -> element-wise (Hadamard) gate
        line 10 return o                          -> transpose back to (B,4C,L1)

    -----------------------------------------------------------------------
    L1 x L1 scaling caveat
    -----------------------------------------------------------------------
    The attention matrix is ``L1 x L1`` per head. At *trial* length this is not
    actually a problem -- measured at 4.05 GiB for C=22, L=1000, B=64. It bites
    on full recordings, where the encoder emits ``L1 ~ 142k``: the score matrix
    alone is ~2e10 entries per head, and even the ``torch.eye(L1)`` diagonal
    mask is a 20 GiB bool tensor. Since self-supervised pretraining on
    continuous windows is exactly the long-sequence regime, ``attn_impl="flex"``
    exists to make it reachable.

    -----------------------------------------------------------------------
    ``attn_impl`` and ``local_window``
    -----------------------------------------------------------------------
    ``attn_impl="dense"`` (default) is the line-for-line reference above and is
    what every number in ``results/eeg_bench/`` was produced with.

    ``attn_impl="flex"`` routes the same computation through
    ``torch.nn.attention.flex_attention``, which fuses masking into the kernel
    and never materializes ``L1 x L1``. This is why FlashAttention-2's standard
    API is *not* the right tool here: Algorithm 2 needs an arbitrary
    (diagonal-excluding) mask, which that API cannot express, whereas a
    ``mask_mod`` states it directly. Memory drops from O(L^2) to O(L).

    ``local_window=w`` additionally restricts attention to ``|i - j| <= w``,
    which drops *compute* to O(L*w) as well -- under flex, whole score blocks
    outside the band are skipped rather than computed and discarded.

    A note on what the band is and is not for: it is a memory/compute measure,
    not a correction to the diagonal mask. Removing the diagonal at ``L1~180``
    withholds 1/180 of the softmax mass and leaves ``i +/- 1, i +/- 2, ...``
    fully attendable, so it does not make attention behave like a high-pass
    filter. Banding is a real cost win on long sequences and a real inductive
    bias; it is not a bug fix, and at trial length it should be ablated on the
    same footing as anything else rather than assumed beneficial.
    """

    def __init__(self, input_channels, embed_dim, ffn_dim, num_heads,
                 attn_impl="dense", local_window=0):
        # input from eeg encoder
        super(GlobalAttentionModule, self).__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        if attn_impl not in ("dense", "flex"):
            raise ValueError(f"attn_impl must be 'dense' or 'flex', got {attn_impl!r}")
        self.attn_impl = attn_impl
        # local_window=0 means unrestricted (full sequence, diagonal excluded).
        if local_window < 0:
            raise ValueError(f"local_window must be >= 0, got {local_window}")
        self.local_window = int(local_window)
        # Cache of compiled BlockMasks, keyed by (L1, device). Building one is not free
        # and L1 is constant across a training run, so rebuilding it every forward would
        # dominate the saving it exists to provide.
        self._block_mask_cache = {}

        # Multi-head split: E is partitioned into `num_heads` heads of size
        # head_dim = E / num_heads. RoPE requires head_dim be even.
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim % 2 == 0, (
            f"head_dim ({self.head_dim}) must be even for RoPE (rotates dim pairs)"
        )

        # Algorithm 2, line 1: LayerNorm over the embedding dim E (= 4C),
        # NOT over input_channels. This is the per-token feature vector.
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # Algorithm 2, line 2: three SEPARATE bias-free q/k/v projections.
        self.q_proj_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Algorithm 2, line 4: sigmoid gating branch, Linear WITH bias.
        self.gate_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Algorithm 2, line 8: output projection, Linear WITH bias.
        self.out_proj_linear = nn.Linear(self.embed_dim, self.embed_dim)

        # Algorithm 2, line 5: scale is 1/sqrt(head_dim) (NOT sqrt(head_dim)).
        # Stored as a plain python float so it multiplies cleanly and never
        # crashes (the old `torch.from_numpy(np.sqrt(int))` was both wrong in
        # value and a runtime error on a 0-d numpy scalar).
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Algorithm 2, line 7: residual FFN = Linear -> ReLU -> Linear.
        self.fnn = nn.Sequential(
            nn.Linear(self.embed_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, self.embed_dim)
        )

        # Algorithm 2, line 6 / line 7 post-residual LayerNorms.
        self.norm_attn = nn.LayerNorm(self.embed_dim)
        self.norm_ffn = nn.LayerNorm(self.embed_dim)

        # RoPE frequency base (theta). Buffer of inverse frequencies for the
        # head_dim/2 rotation pairs; registered so it moves with .to(device).
        self.rope_theta = 10000.0
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------
    # RoPE (Algorithm 2, line 3): hand-implemented rotary position embedding.
    # Applied per head, over the L1 position axis, to q and k ONLY (not v).
    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_half(x):
        # Splits the last dim in half and rotates: (x1, x2) -> (-x2, x1).
        # This is the standard RoPE "rotate_half" convention pairing
        # dimension i with dimension i + head_dim/2.
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x):
        # x: (B, H, L1, head_dim). Build cos/sin for positions 0..L1-1 and
        # rotate each head's head_dim vector by its position-dependent angle.
        seq_len = x.shape[-2]
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        # (L1, head_dim/2) outer product of positions and inverse frequencies.
        freqs = torch.outer(positions, self.inv_freq)
        # Duplicate to full head_dim so the two halves share angles (matches
        # rotate_half pairing): (L1, head_dim).
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :].to(x.dtype)  # (1,1,L1,head_dim)
        sin = emb.sin()[None, None, :, :].to(x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)

    # ------------------------------------------------------------------
    # Algorithm 2, line 5: masked scaled-dot-product attention.
    #
    # Diagonal masking removes self-attention. The paper writes
    # mask(A) = A - diag(A) ("remove diagonal elements to avoid
    # self-attention"). Literally subtracting the diagonal only zeroes the
    # pre-softmax logit, which does NOT remove the self-position from the
    # softmax denominator. The faithful realization of the intent -- that a
    # token contributes ~0 weight to itself -- is to exclude the diagonal
    # BEFORE softmax so it drops out of both numerator and normalization.
    # Both implementations below do exactly that; they differ only in whether
    # the L1 x L1 score matrix is ever materialized.
    # ------------------------------------------------------------------
    def _attend(self, q, k, v, L1):
        # A single token with the diagonal excluded has no keys left, so every
        # logit in its row is -inf and softmax returns NaN. Silently emitting
        # NaN here would surface much later as a dead loss, so make it explicit.
        if L1 < 2:
            raise ValueError(
                f"sequence length L1={L1} leaves no keys once the diagonal is "
                "excluded; Algorithm 2 needs at least 2 tokens"
            )

        if self.attn_impl == "flex":
            return self._attend_flex(q, k, v, L1)

        # ---- dense reference path: materializes (B, H, L1, L1) ----
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        neg_inf = torch.finfo(scores.dtype).min

        idx = torch.arange(L1, device=scores.device)
        block = idx[:, None] == idx[None, :]  # diagonal
        if self.local_window:
            # Outside the band is masked as well.
            block = block | ((idx[:, None] - idx[None, :]).abs() > self.local_window)
        scores = scores.masked_fill(block, neg_inf)

        attn = torch.softmax(scores, dim=-1)  # (B, H, L1, L1)
        return torch.matmul(attn, v)  # (B, H, L1, head_dim)

    def _attend_flex(self, q, k, v, L1):
        """O(L) memory equivalent of ``_attend``'s dense path.

        Compiling *both* entry points is load-bearing, not an optimization. Eager
        ``flex_attention`` is a *reference* implementation that materializes the score
        matrix, and eager ``create_block_mask`` evaluates ``mask_mod`` over the full
        ``L1 x L1`` grid. Measured at L1=32768: the uncompiled mask build alone peaks at
        10240 MiB versus 2 MiB compiled. Left uncompiled, this path is *worse* than the
        dense one it replaces -- 3360 MiB vs 2140 MiB at L1=8192.
        """
        key = (L1, q.device)
        block_mask = self._block_mask_cache.get(key)
        if block_mask is None:
            window = self.local_window

            def mask_mod(b, h, q_idx, kv_idx):
                not_self = q_idx != kv_idx
                if window:
                    return not_self & ((q_idx - kv_idx).abs() <= window)
                return not_self

            block_mask = _compiled_flex("create_block_mask")(
                mask_mod, B=None, H=None, Q_LEN=L1, KV_LEN=L1, device=q.device,
            )
            self._block_mask_cache[key] = block_mask

        return _compiled_flex("flex_attention")(
            q, k, v, block_mask=block_mask, scale=self.scale
        )

    def forward(self, x):
        # ----- shape handling: accept (B, 4C, L1) or unbatched (4C, L1) -----
        squeeze_back = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (4C, L1) -> (1, 4C, L1)
            squeeze_back = True

        in_shape = x.shape
        B, feat_channels, L1 = x.shape
        assert feat_channels == self.embed_dim, (
            f"input feature-channels ({feat_channels}) must equal embed_dim "
            f"(E=4C={self.embed_dim})"
        )

        # Encoder emits (B, 4C, L1); attention treats L1 as tokens and 4C as the
        # embedding E, so transpose to (B, L1, E).
        z = x.transpose(1, 2)  # (B, L1, E)

        # --- Algorithm 2, line 1: input LayerNorm over E ---
        z_bar = self.layer_norm(z)  # (B, L1, E)

        # --- Algorithm 2, line 2: bias-free q/k/v projections ---
        q = self.q_proj_linear(z_bar)  # (B, L1, E)
        k = self.k_proj_linear(z_bar)
        v = self.v_proj_linear(z_bar)

        # Split E into heads: (B, L1, E) -> (B, H, L1, head_dim).
        def split_heads(t):
            return t.view(B, L1, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)  # (B, H, L1, head_dim)
        k = split_heads(k)
        v = split_heads(v)

        # --- Algorithm 2, line 3: RoPE on q and k only, per head, over L1 ---
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # --- Algorithm 2, line 4: sigmoid gate from z_bar (Linear WITH bias) ---
        g = torch.sigmoid(self.gate_linear(z_bar))  # (B, L1, E)

        # --- Algorithm 2, line 5: row-masked multi-head attention ---
        attn_out = self._attend(q, k, v, L1)  # (B, H, L1, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L1, self.embed_dim)

        # --- Algorithm 2, line 6: residual with z_bar, then LayerNorm ---
        z = self.norm_attn(attn_out + z_bar)  # (B, L1, E)

        # --- Algorithm 2, line 7: residual FFN, then LayerNorm ---
        z = self.norm_ffn(z + self.fnn(z))  # (B, L1, E)

        # --- Algorithm 2, line 8: output projection (Linear WITH bias) ---
        o = self.out_proj_linear(z)  # (B, L1, E)

        # --- Algorithm 2, line 9: element-wise sigmoid gating ---
        o = o * g  # (B, L1, E)

        # --- Algorithm 2, line 10: return, transposed back to (B, 4C, L1) ---
        out = o.transpose(1, 2).contiguous()  # (B, E=4C, L1)
        assert out.shape == in_shape, (
            f"output shape {tuple(out.shape)} != input shape {tuple(in_shape)}"
        )

        if squeeze_back:
            out = out.squeeze(0)  # (4C, L1)
        return out
    
if __name__ == "__main__":
    # Synthetic rather than file-backed: the previous smoke test loaded a hardcoded
    # .npy path, so this module could not even be imported where that file was absent.
    C, S = 22, 750  # BCI IV 2a: 22 electrodes, 3 s at 250 Hz
    eeg_test = torch.randn(1, C, S)
    print(f"EEG shape: {tuple(eeg_test.shape)}, dtype: {eeg_test.dtype}")

    for label, kw in (
        ("as published", {}),
        ("branch2 destrided", dict(branch2_stride=1)),
        ("+ spatial mixing", dict(branch2_stride=1, spatial_filter=C)),
        ("+ spectral branch", dict(branch2_stride=1, spatial_filter=C, spectral=True)),
    ):
        enc = EEGEncoder(C, output_channels_per_group=2, kernel_size=32, **kw).eval()
        with torch.no_grad():
            b2 = enc.encoder_two(enc.spatial(eeg_test) if enc.spatial else eeg_test)
            out = enc(eeg_test)
        print(f"  {label:<20} -> {tuple(out.shape)}  E={enc.embed_dim:<4} "
              f"branch2 timepoints={b2.shape[-1]} (stretched "
              f"{out.shape[-1] / b2.shape[-1]:.1f}x)")

    # The long-sequence path. A full recording gives L1 ~ 142k, where the dense
    # L1 x L1 score matrix -- and even its torch.eye diagonal mask -- exceed GPU memory.
    enc = EEGEncoder(C, output_channels_per_group=2, kernel_size=32, branch2_stride=1)
    attn_kw = dict(input_channels=C, embed_dim=enc.embed_dim, ffn_dim=256, num_heads=4)
    with torch.no_grad():
        short = enc(eeg_test)
    dense = GlobalAttentionModule(**attn_kw, attn_impl="dense").eval()
    flex = GlobalAttentionModule(**attn_kw, attn_impl="flex").eval()
    flex.load_state_dict(dense.state_dict())
    with torch.no_grad():
        agreement = (dense(short) - flex(short)).abs().max().item()
    print(f"\nAttention shape: {tuple(dense(short).shape)}")
    print(f"dense vs flex max abs difference: {agreement:.2e}")

class fMRIEncoder(nn.Module):
    def __init__(self, ):
        super(fMRIEncoder, self).__init__()
