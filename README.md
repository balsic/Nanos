# Temporal Multi-scale Parameter-Optimized Encoder

Electroencephalography (EEG) decoding under inter-subject distribution shift is constrained by heterogeneous spatial layouts, subject-specific channel statistics, and temporally localized neural dynamics. We introduce an EEG encoder that explicitly factorizes temporal and spatial modeling through parallel dilated temporal convolutions at multiple receptive-field scales, followed by a learned channel-mixing operator and an optional gated attention mechanism for adaptive temporal feature aggregation. The architecture operates directly on minimally processed multichannel EEG without subject-specific architectural modifications. We conduct a controlled benchmark against 13 reference architectures using an identical optimization, preprocessing, and evaluation protocol across seven evaluation cells comprising five motor-imagery settings, one P300 setting, and one SSVEP setting spanning three datasets, cross- and within-subject protocols, and varying class cardinalities. The evaluation comprises 781 independent fold-units with zero failed training runs; label-permutation controls are included in every cell to establish empirical chance baselines. Across heterogeneous subject and paradigm shifts, the proposed multi-scale encoder provides competitive decoding performance while maintaining a comparatively simple temporal--spatial inductive bias. These results indicate that explicit multi-scale temporal receptive fields coupled with learned spatial mixing provide a robust basis for cross-subject EEG representation learning across substantially different BCI paradigms.

# Reproducing Results

1. Prepare dataset splits
```python
python -m eegbench.prepare --dataset cho2017 --paradigm leftright --tmin 0.5 --tmax 3.0
```
2. Run reference and architecture sweeps
```python
python -m eegbench.bench run sweeps/reference.json --results results/reference
python -m eegbench.bench run sweeps/reshape.json   --results results/reshape
```
3. Aggregate and build a report against TEMPO
```python
python -m eegbench.bench report results/reference --reference ours
```
