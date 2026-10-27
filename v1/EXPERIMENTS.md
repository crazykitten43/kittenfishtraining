# Perspective diagnosis — 2026-10-27

Dataset: `crazykitten43/kittenfishdata/positions_100k.csv.gz`, Git blob
`f2fcbf271b144dbd16a169418b39c81c053d8256`.
100,000 rows; deterministic existing split: 80,128 train / 19,872 validation.
No duplicate positions found. All scores clipped to +/-2000cp by the loader.

## Confirmed input convention

The generator in `crazykitten43/kittenfishfullstack/PositionGeneration.py`
(blob `72dff5232ade1733e62ece4847e84a8830d8e153`) uses:

```python
score = info["score"].pov(board.turn).score(mate_score=10000)
```

The stored CSV scores are side-to-move, whereas the C++ TSV loader expects
White perspective and converts to side-to-move internally. Earlier inline
CSV-to-TSV instructions failed to convert first, reversing all Black-to-move
training targets. Correct this in the converter; do not change the loader and
converter at the same time or the sign correction will be applied twice.

As an independent check, raw CSV labels correlate with White material at
+0.743 for White-to-move positions and -0.746 for Black-to-move positions.

## Experiments, unchanged architecture

| Run | Validation float MSE (cp squared) | Validation float MAE (cp) |
| --- | ---: | ---: |
| Zero baseline | 453727 | 419.94 |
| Material baseline (correct convention) | 216023 | 270.48 |
| Corrected labels, LR 0.0001, epoch 1 | 408012 | 404.38 |
| Corrected labels, LR 0.0001, epoch 2 | 373010 | 382.78 |
| Corrected labels, LR 0.0001, epoch 3 | 343488 | 366.34 |
| Corrected labels, LR 0.0001, total epoch 10 | 215605 | 287.88 |
| Corrected labels, LR 0.001, best MSE in 10 epochs (epoch 9) | 209018 | 291.57 |
| Corrected labels, LR 0.0001, best MSE in 20 epochs (epoch 19) | 194411 | 272.84 |

The 0.0001 run consisted of 3 epochs, a 7-epoch continuation, and a 10-epoch
continuation. Shuffling restarts at seed 42 on each continuation, so a single
20-epoch run will not produce identical weights. The 0.001 trial started fresh.
Validation informed these choices; this is development validation, not an
independent final test set or evidence of playing strength.

At the best float checkpoint in the 20-epoch experiment, quantized validation
MSE was 196644 cp squared and MAE 277.54cp. These numbers do not justify claiming
that the network beats material-only evaluation on MAE. It does improve MSE.
The final epoch's training sample MAE was 208.98cp versus validation 278.46cp,
so simply continuing many more epochs is not the next priority. Data volume,
feature sharing, regularization, and quantization precision remain candidates
for controlled follow-up experiments. No architecture change is part of this fix.

Checks: incremental-vs-rebuilt accumulators; finite-difference gradients for
feature, bias and output weights; SGD loss reduction; quantized roundtrip;
converter perspective/grouping/deduplication tests; AddressSanitizer and UBSan
selftest (LeakSanitizer disabled because this runtime prevents process inspection).
