"""Dataset specifications: what each corpus is, and what it is allowed to claim.

A spec is a *contract*, not a convenience. Three fields in particular are load-bearing and
each has a documented failure mode if it drifts from the data:

``events``
    Defines the integer label mapping, and the order is fixed here rather than derived
    from the data. Deriving it (``LabelEncoder`` sorts alphabetically) silently remaps
    class indices for any subject missing a class, so class *k* would mean different
    things in different rows of a results table.

``n_channels``
    Validated against what the paradigm actually returns. This is not paranoia: MOABB's
    channel selection depends on the paradigm and the montage, and a spec that disagrees
    with the data yields a plausible tensor rather than an error.

``min_post_cue``
    The shortest usable window, in seconds, measured from the cue. It is what decides
    whether a dataset may join a pooled cohort: pooling needs one common ``n_times``, and
    a dataset that cannot supply the requested window must be refused rather than padded.

Subject and trial counts below marked "verified" were read from prepared caches on this
machine, not copied from the papers.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DatasetSpec", "REGISTRY", "get", "names", "leftright_capable"]


@dataclass(frozen=True)
class DatasetSpec:
    """One corpus, and the constraints under which it may be used."""

    key: str
    moabb_name: str
    n_subjects: int
    #: ``None`` means "never measured". The channel check exists to catch a count that
    #: *changed* from what this project observed; for a corpus nobody here has read yet
    #: there is no expectation to violate, so the first preparation measures it, records
    #: it in the sidecar, and prints the registry line to paste back. Strictness applies
    #: from the second run onward, which is when it can actually catch something.
    n_channels: int | None
    native_sfreq: float
    n_sessions: int
    events: tuple[str, ...]
    #: Longest window, in seconds from the cue, that every trial in the corpus supports.
    #: The shared pooling contract needs 3.0; anything below that cannot join it.
    min_post_cue: float
    #: Default analysis window when the caller does not pass one.
    tmin: float = 0.5
    tmax: float = 3.0
    trials_per_session: int | None = None
    #: ``True`` once a prepared cache for this dataset has been observed on disk.
    verified: bool = False
    note: str = ""

    def supports(self, tmin: float, tmax: float) -> bool:
        """Can this corpus supply the window ``[tmin, tmax]`` after the cue?"""
        return tmax <= self.min_post_cue + 1e-9 and tmin >= 0.0

    def has_leftright(self) -> bool:
        return "left_hand" in self.events and "right_hand" in self.events


def _s(**kw) -> DatasetSpec:
    return DatasetSpec(**kw)


# ---------------------------------------------------------------------------------------
# Prepared and verified on this machine. Subject / channel / trial counts below were read
# out of `data/prepared/**/sub-*.json`, so they describe what is actually loadable rather
# than what the publications report.
# ---------------------------------------------------------------------------------------
_VERIFIED: tuple[DatasetSpec, ...] = (
    _s(
        key="bnci2014_001", moabb_name="BNCI2014_001", n_subjects=9, n_channels=22,
        native_sfreq=250.0, n_sessions=2,
        events=("left_hand", "right_hand", "feet", "tongue"),
        min_post_cue=3.5, tmin=0.5, tmax=3.5, trials_per_session=288, verified=True,
        note="BCI Competition IV 2a. The reference cell: 2 sessions allow the "
             "subject-dependent holdout protocol, which most published numbers use.",
    ),
    _s(
        key="bnci2014_004", moabb_name="BNCI2014_004", n_subjects=9, n_channels=3,
        native_sfreq=250.0, n_sessions=5, events=("left_hand", "right_hand"),
        min_post_cue=3.0, verified=True,
        note="3 electrodes (C3/Cz/C4). Contributes 6,520 trials but only 3 of any union "
             "montage's columns -- the extreme case for montage dilution.",
    ),
    _s(
        key="cho2017", moabb_name="Cho2017", n_subjects=52, n_channels=64,
        native_sfreq=512.0, n_sessions=1, events=("left_hand", "right_hand"),
        min_post_cue=3.0, verified=True,
        note="52 subjects, 10,520 trials. The primary cross-subject cell: large enough "
             "for 10 folds, which is where a paired test has power.",
    ),
    _s(
        key="liu2024", moabb_name="Liu2024", n_subjects=50, n_channels=29,
        native_sfreq=500.0, n_sessions=1, events=("left_hand", "right_hand"),
        min_post_cue=3.0, verified=True,
        note="29 channels as returned by the paradigm, not the 32 the paper implies -- "
             "the spec/data mismatch check is what caught this.",
    ),
    _s(
        key="schirrmeister2017", moabb_name="Schirrmeister2017", n_subjects=14,
        n_channels=128, native_sfreq=500.0, n_sessions=1,
        events=("left_hand", "right_hand"), min_post_cue=3.0, verified=True,
        note="High-density. 6,742 trials from 14 subjects: the depth-rich, breadth-poor "
             "corner of the cohort.",
    ),
    _s(
        key="weibo2014", moabb_name="Weibo2014", n_subjects=10, n_channels=60,
        native_sfreq=200.0, n_sessions=1, events=("left_hand", "right_hand"),
        min_post_cue=3.0, verified=True,
    ),
    _s(
        key="zhou2016", moabb_name="Zhou2016", n_subjects=4, n_channels=14,
        native_sfreq=250.0, n_sessions=3, events=("left_hand", "right_hand"),
        min_post_cue=3.0, verified=True,
    ),
    _s(
        key="physionet_mi", moabb_name="PhysionetMI", n_subjects=109, n_channels=64,
        native_sfreq=160.0, n_sessions=1,
        events=("left_hand", "right_hand", "hands", "feet"),
        min_post_cue=3.0, verified=True,
        note="109 subjects -- the largest subject count on disk. Currently prepared only "
             "as 4-class; it defines left_hand/right_hand, so preparing it under the "
             "left/right contract is the single largest cheap addition to the cohort.",
    ),
)

# ---------------------------------------------------------------------------------------
# Raw data present, prepared cache empty. Each of these needs one `prepare` run; the
# reason each is currently missing is recorded once known, so a failure is not
# re-diagnosed every time somebody notices the gap.
# ---------------------------------------------------------------------------------------
# Every ``n_channels`` below was *measured* by running the paradigm on subject 1, not
# taken from a publication. Four of them contradicted the published figure, and since the
# preparation step validates the spec against the data, a wrong literal here is a hard
# failure that reads like a broken dataset -- which is exactly how these seven came to be
# recorded as "unpreparable" when five of them prepare fine.
_UNPREPARED: tuple[DatasetSpec, ...] = (
    _s(key="grossewentrup2009", moabb_name="GrosseWentrup2009", n_subjects=10,
       n_channels=128, native_sfreq=500.0, n_sessions=1,
       events=("left_hand", "right_hand"), min_post_cue=3.0,
       note="Unblocked via eegbench._compat.patch_pymatreader_opaque (pymatreader 1.2.3 "
            "+ scipy>=1.15 raise on an array-valued comparison reading EEGLAB .set "
            "files). Prepares cleanly: 300 balanced trials, 128 channels, ~52 s/subject, "
            "zero download. BUT SINGLE-DATASET ONLY -- MOABB returns its montage as the "
            "bare numerals '1'..'128' rather than electrode names, so it cannot join a "
            "union montage. store._assert_montages_are_comparable refuses it."),
    _s(key="wairagkar2018", moabb_name="Wairagkar2018", n_subjects=14, n_channels=19,
       native_sfreq=1000.0, n_sessions=1, events=("left_hand", "right_hand"),
       min_post_cue=3.0,
       note="19 channels measured, not the 32 published. The 688 MB archive on disk is "
            "intact and holds all 14 subjects; MOABB's loader passes a str where "
            "safe_extract_zip wants a ZipFile, so extracting the archive in place "
            "side-steps the bug entirely. ~12 s per subject."),
    _s(key="kumar2024", moabb_name="Kumar2024", n_subjects=18, n_channels=22,
       native_sfreq=250.0, n_sessions=6, events=("left_hand", "right_hand"),
       min_post_cue=3.0,
       note="22 channels measured, not 32. Raw is complete for all 18 subjects and "
            "needs no download; ~91 s per subject. The largest free block of subjects."),
    _s(key="forenzo2023", moabb_name="Forenzo2023", n_subjects=25, n_channels=64,
       native_sfreq=256.0, n_sessions=5, events=("left_hand", "right_hand"),
       min_post_cue=3.0,
       note="64 channels measured, not 32; real electrode names, so it pools correctly. "
            "MOABB's data_path globs Subject{N}/*.mat where the zip extracts to "
            "Subject{N}/publicData/*.mat, so it re-downloads 3.39 GB per subject even "
            "when the files are present. Fix the glob before pulling this one."),
    _s(key="zhou2020", moabb_name="Zhou2020", n_subjects=20, n_channels=41,
       native_sfreq=250.0, n_sessions=7, events=("left_hand", "right_hand"),
       min_post_cue=3.0,
       note="41 channels measured, not 64, and only subject 1 of 20 is on disk (~21 GB "
            "to complete). MOABB reports its channels as 'EEG1'..'EEG41' rather than "
            "electrode names, so it CANNOT join a union montage -- it would open 41 "
            "private columns matching no other corpus. Usable single-dataset only, "
            "until the names are recovered from the source documentation."),
    _s(key="brandl2020", moabb_name="Brandl2020", n_subjects=16, n_channels=63,
       native_sfreq=500.0, n_sessions=1, events=("left_hand", "right_hand"),
       min_post_cue=3.0,
       note="BLOCKED, not merely unprepared. The two files on disk are 1,306-byte HTML "
            "error pages saved under .mat names, and the upstream DSpace URL still "
            "serves HTML today -- so re-downloading cannot help. Needs a new source."),
    _s(key="shin2017a", moabb_name="Shin2017A", n_subjects=29, n_channels=30,
       native_sfreq=200.0, n_sessions=3, events=("left_hand", "right_hand"),
       min_post_cue=3.0,
       note="BLOCKED: licence gate and zero raw bytes on disk."),
)

# ---------------------------------------------------------------------------------------
# Not downloaded. Subject-rich, and therefore high value: at fixed data volume, subject
# diversity has been measured on this cohort as worth roughly 3x extra trials, so a
# corpus is ranked by how many *people* it adds rather than how many trials.
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# Present in MOABB, subject-rich, and never touched by this project. Channel counts are
# deliberately ``None``: taking them from a publication is what produced four wrong
# literals above, so they are measured on first preparation instead.
#
# For scale: MOABB carries ~30 corpora defining left_hand/right_hand motor imagery,
# totalling ~875 subjects. The cohort assembled here is a small fraction of that, and the
# binding constraint is heterogeneity rather than availability -- see PLAN.md section 0.
# ---------------------------------------------------------------------------------------
_UNSURVEYED: tuple[DatasetSpec, ...] = (
    _s(key="stieger2021", moabb_name="Stieger2021", n_subjects=62, n_channels=None,
       native_sfreq=1000.0, n_sessions=1,
       events=("left_hand", "right_hand", "hands", "feet"), min_post_cue=3.0,
       note="62 subjects, the largest MI corpus this project has never looked at. Its "
            "default interval is [0, 3], so the 0.5-3.0 s contract is exactly available "
            "with no margin."),
    _s(key="yang2025", moabb_name="Yang2025", n_subjects=51, n_channels=None,
       native_sfreq=500.0, n_sessions=1, events=("left_hand", "right_hand"),
       min_post_cue=4.0, tmin=1.5, tmax=4.5,
       note="Default interval [1.5, 5.5] -- the cue-relative window starts LATE. Verify "
            "what t=0 means here before pooling it on the shared 0.5-3.0 s contract; a "
            "window offset relative to the cue is silent and would misalign every trial."),
    _s(key="hefmiich2025", moabb_name="HefmiIch2025", n_subjects=37, n_channels=None,
       native_sfreq=500.0, n_sessions=1, events=("left_hand", "right_hand"),
       min_post_cue=3.0, note="Interval [0, 10]; ample window."),
    _s(key="guttmannflury2025_mi", moabb_name="GuttmannFlury2025_MI", n_subjects=31,
       n_channels=None, native_sfreq=500.0, n_sessions=1,
       events=("left_hand", "right_hand"), min_post_cue=3.0),
    _s(key="chang2025", moabb_name="Chang2025", n_subjects=28, n_channels=None,
       native_sfreq=500.0, n_sessions=1, events=("left_hand", "right_hand"),
       min_post_cue=3.0),
)

_REMOTE: tuple[DatasetSpec, ...] = (
    _s(key="dreyer2023", moabb_name="Dreyer2023", n_subjects=87, n_channels=27,
       native_sfreq=512.0, n_sessions=1, events=("left_hand", "right_hand"),
       min_post_cue=3.0,
       note="The best value available by a wide margin: 87 subjects for ~5.6 GB of "
            "download (~14 GB on disk), not licence-gated. Prefer this over the "
            "60-subject Dreyer2023A sub-cohort -- the B and C cohorts share A's "
            "27-channel montage, so taking all three costs nothing in montage terms "
            "and adds 27 people."),
    _s(key="lee2019_mi", moabb_name="Lee2019_MI", n_subjects=54, n_channels=62,
       native_sfreq=1000.0, n_sessions=2, events=("left_hand", "right_hand"),
       min_post_cue=3.0,
       note="54 subjects for ~65 GB -- an order of magnitude worse per subject than "
            "Dreyer2023, so do it second. Two cautions: MOABB's Lee2019_MI takes "
            "`test_run` and defaults it to False, so constructing it with no arguments "
            "silently takes half the trials (10,800 of 21,600); and n_channels=62 comes "
            "from the same metadata-constant source that was wrong for liu2024, so "
            "smoke-test one subject before committing the download."),
)


# ---------------------------------------------------------------------------------------
# Stimulus-driven paradigms. These are NOT motor imagery and must not be pooled with it:
# different physics, different band, different window, different scalp focus. They are here
# so the encoder can be tested outside the one paradigm it was tuned on.
# ---------------------------------------------------------------------------------------
_STIMULUS: tuple[DatasetSpec, ...] = (
    _s(key="bnci2014_009", moabb_name="BNCI2014_009", n_subjects=10, n_channels=16,
       native_sfreq=256.0, n_sessions=3, events=("NonTarget", "Target"),
       min_post_cue=0.8, tmin=0.0, tmax=0.8,
       note="P300 speller, 3 sessions. Roughly 1:5 target:non-target, so report balanced "
            "accuracy and AUC -- a constant NonTarget predictor scores ~83% plain."),
    _s(key="bnci2014_008", moabb_name="BNCI2014_008", n_subjects=8, n_channels=None,
       native_sfreq=256.0, n_sessions=1, events=("NonTarget", "Target"),
       min_post_cue=1.0, tmin=0.0, tmax=0.8,
       note="P300, ALS patients. Single session."),
    _s(key="kalunga2016", moabb_name="Kalunga2016", n_subjects=12, n_channels=None,
       native_sfreq=256.0, n_sessions=1, events=("13", "17", "21", "rest"),
       min_post_cue=2.0, tmin=0.0, tmax=2.0,
       note="SSVEP, occipital, includes a 'rest' class so 4-way. BLOCKED: after "
            "band-passing 7-45 Hz its median amplitude is 1.99e-3 V (~2 mV), which is "
            "three orders above plausible scalp EEG and trips the volt-range guard, "
            "despite unit_factor=1e6 like every other corpus. Its unit convention needs "
            "establishing before the data can be trusted; it is not a preparation bug."),
    _s(key="nakanishi2015", moabb_name="Nakanishi2015", n_subjects=9, n_channels=8,
       native_sfreq=256.0, n_sessions=1,
       events=("9.25", "11.25", "13.25", "9.75", "11.75", "13.75",
               "10.25", "12.25", "14.25", "10.75", "12.75", "14.75"),
       min_post_cue=4.0, tmin=0.5, tmax=3.5,
       note="SSVEP, 12 stimulation frequencies, 8 electrodes. The high-SNR case: a "
            "failure here means something is wrong, not merely noisy."),
)

REGISTRY: dict[str, DatasetSpec] = {
    s.key: s for s in (*_VERIFIED, *_UNPREPARED, *_UNSURVEYED, *_REMOTE, *_STIMULUS)
}


def get(key: str) -> DatasetSpec:
    if key not in REGISTRY:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def names(*, verified_only: bool = False) -> list[str]:
    return sorted(k for k, v in REGISTRY.items() if v.verified or not verified_only)


def leftright_capable(tmin: float = 0.5, tmax: float = 3.0,
                      *, verified_only: bool = True) -> list[str]:
    """Datasets that can supply left/right trials over ``[tmin, tmax]``.

    This is the membership test for a pooled cohort. It is deliberately a *predicate over
    the spec* rather than a hand-maintained list, so adding a corpus cannot accidentally
    admit one that lacks the window or the events.
    """
    return sorted(
        k for k, s in REGISTRY.items()
        if s.has_leftright() and s.supports(tmin, tmax)
        and (s.verified or not verified_only)
    )
