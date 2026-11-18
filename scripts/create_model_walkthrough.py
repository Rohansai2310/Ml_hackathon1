"""Create a notebook that explains and reproduces the four completed CV experiments.

The default cells are executed as Python and capture real output from saved files.
They do not retrain models or fabricate training evidence.
"""
import contextlib
import io
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
            "execution_count": None, "outputs": []}


cells = [
markdown('''# PathSnap: how the four models were run

This walkthrough shows the **actual recorded five-fold cross-validation experiments** for Logistic Regression, Gaussian Naive Bayes, Decision Tree and Random Forest.

Default mode reads the saved experiment files. It does **not** claim to retrain models. Change `RETRAIN_MODELS` below to rerun selected models on the complete original training data; new outputs go into a separate directory. Random Forest originally took about 50 minutes for five fits, excluding scoring and output overhead.

These are classifier-only results against valid derived reference labels. Two unreliable training trips were excluded. AdaBoost, Bayesian tuning, student-testing and HMM evaluation are outside these four recorded CV experiments. Final deployment model files have not been saved by these CV runs.
'''),
code(r'''from pathlib import Path
import sys
import json
import inspect
from datetime import datetime

# Works when opened from either the repository root or its notebooks directory.
ROOT = next((p for parent in [Path.cwd(), *Path.cwd().parents]
             for p in [parent, parent / 'Ml_hackathon1']
             if (p / 'src/pathsnap/train.py').is_file()), None)
if ROOT is None:
    raise RuntimeError('Open this notebook from the Ml_hackathon1 repository')
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathsnap import config
from pathsnap.models import build_model
from pathsnap.train import cross_validate, make_folds, write_json
from pathsnap.training_handoff import load_feature_dir

RUN = ROOT / 'outputs/experiments/run_001'
FEATURES = ROOT / 'outputs/features'
RETRAIN_MODELS = set()  # Example: {'logistic_regression'}; leave empty to show recorded results.
RETRAIN_OUTPUT = ROOT / 'outputs/training' / datetime.now().strftime('walkthrough_%Y%m%d_%H%M%S')
metadata = json.loads((RUN / 'run.json').read_text())
SEED, JOBS = metadata['seed'], metadata['jobs']
print('Repository:', ROOT)
print('Recorded run:', RUN)
print('Random seed:', SEED, '| Random Forest workers:', JOBS)
print('Mode:', 'retrain selected models' if RETRAIN_MODELS else 'read recorded results')
'''),
markdown('''## 1. The input data

Each row represents one GPS observation paired with one candidate road. The label is `1` when that candidate matches the derived reference arc, and `0` otherwise. Only the 22 declared features enter the model; IDs, labels, fold IDs and reference metadata are excluded from features. `label_valid=False` rows are not used for fitting.
'''),
code(r'''schema = json.loads((FEATURES / 'feature_schema.json').read_text())
feature_names = schema['feature_columns']
for partition in ['train', 'student_test']:
    candidates_file = pq.ParquetFile(FEATURES / f'{partition}_candidates.parquet')
    points_file = pq.ParquetFile(FEATURES / f'{partition}_points.parquet')
    print(f'{partition}: {candidates_file.metadata.num_rows:,} candidate rows, '
          f'{points_file.metadata.num_rows:,} GPS observations')
print('\nFeature columns:')
print(', '.join(feature_names))
preview = next(pq.ParquetFile(FEATURES / 'train_candidates.parquet').iter_batches(batch_size=8)).to_pandas()
print('\nA few real candidate rows (preview only):')
print(preview[['trip_id', 'point_idx', 'arc_id', 'dist_m', 'heading_diff_deg', 'label', 'label_valid', 'fold']].to_string(index=False))
'''),
markdown('''## 2. Keeping whole trips together

All four models used the same five existing folds. A validation trip never contributes rows to the corresponding training fold. Fold `-1` marks trips excluded by the upstream reference-quality rule. The split is by trip, not by candidate row or GPS observation.
'''),
code(r'''assignments = pd.read_csv(RUN / 'fold_assignments.csv', dtype={'trip_id': 'string'})
assert assignments.trip_id.is_unique
assert set(assignments.validation_fold) - {-1} == set(range(5))
print('Trips by validation fold (-1 means excluded):')
print(assignments.groupby('validation_fold').size().rename('trip_count').to_string())
print('\nExcluded trips:', assignments.loc[assignments.validation_fold.eq(-1), 'trip_id'].tolist())
for fold in range(5):
    training_trips = set(assignments.loc[assignments.validation_fold.ge(0) & assignments.validation_fold.ne(fold), 'trip_id'])
    validation_trips = set(assignments.loc[assignments.validation_fold.eq(fold), 'trip_id'])
    assert training_trips.isdisjoint(validation_trips)
print('Verified: no training/validation trip overlap in any fold.')
'''),
markdown('''## 3. The actual shared training loop

For each fold, create a fresh pipeline, fit it on training trips, score candidate roads in held-out trips, and evaluate those predictions. Median imputation is learned on the training fold. Logistic Regression and Gaussian Naive Bayes also receive standard scaling learned on that fold.

The positive-class probability scores each candidate. The highest-scoring candidate per observation becomes the predicted road. Top-1 accuracy includes candidate misses as errors for observations with valid references. Macro F1 and log loss are candidate-level metrics.
'''),
code(r'''print(inspect.getsource(cross_validate))
'''),
code(r'''# Helper for the four sections below. Fitting happens only for names explicitly
# added to RETRAIN_MODELS. Recorded CV results are never overwritten.
training_cache = None


def demonstrate(name):
    global training_cache
    pipeline = build_model(name, seed=SEED, jobs=JOBS)
    print('Pipeline configuration:')
    print(pipeline)
    if name in RETRAIN_MODELS:
        if training_cache is None:
            training, _, _ = load_feature_dir(FEATURES)
            folds, rerun_assignments = make_folds(training, SEED)
            assert rerun_assignments.astype(str).equals(assignments.astype(str))
            training_cache = training, folds
        training, folds = training_cache
        print('\nExecuting fresh training and five-fold CV on full data...')
        summary, scores_by_fold, predictions = cross_validate(name, training, folds, SEED, JOBS)
        directory = RETRAIN_OUTPUT / name
        directory.mkdir(parents=True)
        scores_by_fold.to_csv(directory / 'cv_folds.csv', index=False)
        predictions.to_csv(directory / 'cv_candidate_scores.csv', index=False)
        write_json(directory / 'cv_summary.json', summary)
        write_json(directory / 'configuration.json', {'seed': SEED, 'features': training.features,
                                                    'parameters': pipeline.get_params(deep=True)})
        print('New CV output:', directory)
    else:
        directory = RUN / name
        print('\nLoading recorded results; this cell does not fit the model.')
        summary = json.loads((directory / 'cv_summary.json').read_text())
        scores_by_fold = pd.read_csv(directory / 'cv_folds.csv')
    assert len(scores_by_fold) == 5 and set(scores_by_fold.fold) == set(range(5))
    assert np.isclose(scores_by_fold.top1_accuracy.mean(), summary['top1_accuracy'])
    print('\nFive held-out-fold results:')
    print(scores_by_fold[['fold', 'top1_accuracy', 'macro_f1', 'log_loss', 'training_seconds']].round(4).to_string(index=False))
    print(f"\nMean Top-1: {summary['top1_accuracy']:.2%}")
    print(f"Candidate Macro F1: {summary['macro_f1']:.4f} | Log loss: {summary['log_loss']:.4f}")
    print('\nRecorded candidate scores (first six rows):')
    score_file = directory / 'cv_candidate_scores.csv'
    if not score_file.exists():
        score_file = directory / 'cv_candidate_scores.sample.csv'
    print(pd.read_csv(score_file, nrows=6,
                      dtype={'trip_id': 'string', 'observation_id': 'string', 'candidate_id': 'string'}).to_string(index=False))
    return summary
'''),
]
models = [
('logistic_regression', 'Logistic Regression', 'Median imputation → standard scaling → Logistic Regression. Maximum 2,000 solver iterations, default regularisation C=1, seed from the recorded run.'),
('gaussian_nb', 'Gaussian Naive Bayes', 'Median imputation → standard scaling → GaussianNB. Default variance smoothing is 1e-9.'),
('decision_tree', 'Decision Tree', 'Median imputation → DecisionTreeClassifier. The baseline has no maximum depth and uses the recorded random seed.'),
('random_forest', 'Random Forest', 'Median imputation → RandomForestClassifier with 200 trees, no maximum depth, two workers and the recorded random seed.'),
]
for i, (name, label, description) in enumerate(models, start=4):
    cells.extend([markdown(f'## {i}. {label}\n\n{description}\n'), code(f"{name}_result = demonstrate('{name}')\n")])
cells.extend([
markdown('''## 8. Compare the four completed models

This table is calculated from the outputs shown above. Top-1 is the arithmetic mean across the same five validation folds. A final model must be selected after the required five-model comparison; this notebook does not treat these four baselines as a completed final submission.
'''),
code(r'''results = {
    'Logistic Regression': logistic_regression_result,
    'Gaussian Naive Bayes': gaussian_nb_result,
    'Decision Tree': decision_tree_result,
    'Random Forest': random_forest_result,
}
comparison = pd.DataFrame.from_dict(results, orient='index')
print(comparison[['top1_accuracy', 'macro_f1', 'log_loss', 'training_seconds']].round(4).to_string())
'''),
markdown('''![Recorded four-model comparison](../outputs/experiments/run_001/four_model_comparison.png)

The whiskers show fold standard deviation, not confidence intervals.
'''),
markdown('''## 9. Original terminal evidence

The following lines come from the original local training log. They document the four baseline runs before the local process was interrupted to move AdaBoost to Kaggle. They are historical outputs, not a new training execution by this notebook.
'''),
code(r'''log = (RUN / 'run_001.log').read_text()
first_four_models = log.split('Cross-validating adaboost')[0]
print(first_four_models)
print('\nRecorded environment (first 15 lines):')
print('\n'.join((RUN / 'environment.txt').read_text().splitlines()[:15]))
'''),
markdown('''## 10. Reproduce all four from the terminal

Run the following command from the repository root with the project environment activated. It performs real training in a **new** output directory, runs all four models on the original folds, fits baseline pipelines, and evaluates student-testing. It does not perform five-model selection or Bayesian tuning.

```bash
PYTHONPATH=src python -m pathsnap.train \\
  --feature-dir outputs/features \\
  --output outputs/training/four_model_reproduction_001 \\
  --models logistic_regression gaussian_nb decision_tree random_forest \\
  --baseline-only --jobs 2 --seed 42
```

The original four CV experiments did not retain their temporary fitted fold models. The terminal command additionally saves refitted `pipeline.joblib` artifacts; do not describe those new artifacts as already existing until this command completes.
''')])


def main():
    for index, cell in enumerate(cells):
        cell['id'] = f'four-model-walkthrough-{index}'
    # Execute only the recorded-results mode to populate verifiable notebook output.
    namespace = {'__name__': '__walkthrough__'}
    old_directory = Path.cwd()
    os.chdir(ROOT)
    count = 0
    try:
        for cell in cells:
            if cell['cell_type'] != 'code':
                continue
            count += 1
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                exec(compile(''.join(cell['source']), '<model-walkthrough>', 'exec'), namespace)
            cell['execution_count'] = count
            cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout',
                                'text': captured.getvalue().splitlines(keepends=True)}]
    finally:
        os.chdir(old_directory)
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python (PathSnap)', 'language': 'python', 'name': 'pathsnap'},
        'language_info': {'name': 'python'},
        'pathsnap': {'execution': 'Recorded-results cells executed as Python; no model fitting performed'}},
        'nbformat': 4, 'nbformat_minor': 5}
    target = ROOT / 'notebooks/four_models_walkthrough.ipynb'
    target.write_text(json.dumps(notebook, indent=2, default=str) + '\n')
    print(f'Created and verified {target}')


if __name__ == '__main__':
    main()
