# Decoder tuning alongside client development

The Random Forest is reused without fitting again. `pathsnap.tune_hmm` reads its
original training out-of-fold candidate predictions, checks feature-file hashes
and fold alignment, then tunes decoder settings without opening evaluator data.

```bash
python -m pathsnap.tune_hmm --output outputs/training/hmm_tuning_001
```

The search screens 25 configurations on ten training trips: two trips per fold,
selected deterministically with seed 42. It tests transition scale, probability
versus log-odds emission potentials, and a 0/5/15 m same-arc projection-jitter
tolerance. A small backwards projection within that tolerance is treated as a
stationary noisy observation, not reverse traversal of a directed road.

The three strongest screening candidates and original decoder are checked on
all 78 eligible training trips. Selection maximizes mean fold Top-1, then mean
trip Top-1, with trial order breaking exact ties. Whole trips remain in the
original folds, and all candidate probabilities come from their held-out fold.
Final decoder hyperparameter selection still uses those OOF labels; these scores
are selection metrics, not an independent estimate after decoder selection.

`decoder_configuration.json` is frozen before the ten student-testing trips are
loaded. Student-testing then compares the original decoder, selected decoder and
classifier-only baseline. Trial logs and stage metadata are retained. This is a
bounded search, not a guarantee of improvement or a claim of calibrated emissions.
The optional log-odds transform changes emission strength; it is not a fitted
probability calibration model.

Use a fresh output directory for another run. The local tuning outputs are
ignored by Git while the code and tests can be shared immediately. The selected
configuration can be used by the backend through `PATHSNAP_DECODER_CONFIG` or
by evaluation through `--with-hmm --decoder-config <configuration.json>`. Both
paths require a matching frozen model hash and evaluator-free selection record.

The already examined evaluator set must not be used to rank configurations or
select thresholds. A later repeat benchmark on it is a development comparison,
not a fresh untouched final evaluation.
