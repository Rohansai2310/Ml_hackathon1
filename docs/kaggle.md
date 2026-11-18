# AdaBoost on Kaggle

Use `notebooks/adaboost_kaggle.ipynb` with the generated
`outputs/kaggle/pathsnap_adaboost.zip` dataset. The bundle contains the same
features, point inventories, folds and Python source as the local training run.
It pins the local scikit-learn version; other environment versions are recorded.

The notebook runs **AdaBoost only**: five-fold CV, refitting on eligible training
data, and student-testing predictions. It does not tune or select a final model.
The baseline remains 100 depth-one decision trees, learning rate 0.5, seed 42.

## Upload and run

1. Create a private Kaggle dataset containing `pathsnap_adaboost.zip`.
2. Import `notebooks/adaboost_kaggle.ipynb` into a Kaggle notebook.
3. Attach the dataset and enable Internet for dependency installation.
4. Run all cells using the CPU runtime. Download `adaboost_results.zip`.

Scikit-learn's tree-based AdaBoost does not use a Kaggle GPU. CPU/memory resources
may help, but speed depends on the assigned machine. See the
[scikit-learn GPU FAQ](https://scikit-learn.org/stable/faq.html#will-you-add-gpu-support).
This package is prepared locally; no Kaggle upload or remote execution has been
performed automatically.

## Rebuild the upload bundle

From the repository root, after building features:

```bash
PYTHONPATH=src python -m pathsnap.kaggle_bundle create \
  --feature-dir outputs/features \
  --output outputs/kaggle/pathsnap_adaboost.zip
```

## Bring results back

Extract the downloaded archive into `outputs/kaggle/adaboost_results/`, with
`run.json` directly inside that directory. Then run:

```bash
PYTHONPATH=src python -m pathsnap.kaggle_bundle import-results \
  --external-run outputs/kaggle/adaboost_results \
  --local-run outputs/training/run_001

PYTHONPATH=src python -m pathsnap.train \
  --feature-dir outputs/features \
  --output outputs/training/run_001 \
  --resume --trials 30 --jobs 2
```

Import checks feature-data hashes, seed and exact fold assignments, as well as
the presence of all five fold results and saved baseline artifacts. It refuses
to overwrite nonempty local AdaBoost results. Final model selection and tuning
wait until all five models have their CV results. Resume reuses completed
baseline CV outputs and the imported AdaBoost pipeline/student-test predictions.
Other baseline pipelines are fitted locally after selection/tuning.

All final comparisons still use CV performance; the notebook's student-testing
scores are reporting outputs. Keep the remote execution metadata and environment
record with the report evidence. Training times from Kaggle and the laptop are
hardware-dependent; benchmark inference on the same machine before using a
cross-machine timing tie breaker or claiming one model is faster. Resume retains
the recorded timing values rather than silently treating them as equivalent.

## Local handoff

The previously started local process is scheduled to stop once its Random Forest
CV summary is safely saved. The handoff is recorded in
`outputs/training/run_001/adaboost_deferred.json`. It may create an empty
`adaboost/` directory before interruption; importing into an empty directory is
allowed. No Random Forest fold is deliberately discarded.
