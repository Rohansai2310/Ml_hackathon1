"""Publish compact decoder tuning/validation evidence after training selection finishes."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def report(tuning_dir,archive,evaluator_run=None):
    tuning_dir,archive=Path(tuning_dir),Path(archive)
    run=json.loads((tuning_dir/'run.json').read_text())
    if run['status'] != 'complete':
        raise ValueError('Wait for tuning and student validation to finish')
    selected=json.loads((tuning_dir/'decoder_configuration.json').read_text())
    student=json.loads((tuning_dir/'student_validation.json').read_text())
    rf=json.loads((tuning_dir/'training_rf.json').read_text())
    full=pd.read_csv(tuning_dir/'full_training_trials.csv')
    original=full.loc[full.trial.eq(0)].iloc[0]
    winner=selected['training_metrics']
    archive.mkdir(parents=True,exist_ok=True)
    for path in tuning_dir.iterdir():
        if path.is_file() and path.suffix in {'.json','.csv'}:
            shutil.copy2(path,archive/path.name)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    labels=['RF only','Original HMM','Selected HMM']
    values=[[rf['mean_fold_top1'],original.mean_fold_top1,winner['mean_fold_top1']],
        [student[name]['pooled_top1'] for name in ['random_forest','original_hmm','selected_hmm']]]
    for ax,measure,title in zip(axes,values,['Training OOF (selection)','Student-test (after selection)']):
        bars=ax.bar(labels,[100*v for v in measure],color=['#087e86','#aeb9c5','#d89035'])
        ax.set_ylim(0,105);ax.set_title(title);ax.set_ylabel('Pointwise Top-1 accuracy (%)')
        ax.bar_label(bars,labels=[f'{v:.2%}' for v in measure],padding=4)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(archive/'validation_comparison.png',dpi=160);plt.close(fig)
    lines=['# Training-selected HMM tuning results','',
        'Random Forest was reused without retraining. Evaluator data was not used to screen or select decoder configurations.','',
        '## Measured comparison','',
        '| Pipeline | Training OOF mean-fold Top-1 | Student-test pooled Top-1 |',
        '|---|---:|---:|',
        f"| Random Forest | {rf['mean_fold_top1']:.2%} | {student['random_forest']['pooled_top1']:.2%} |",
        f"| Original HMM | {original.mean_fold_top1:.2%} | {student['original_hmm']['pooled_top1']:.2%} |",
        f"| Selected HMM | {winner['mean_fold_top1']:.2%} | {student['selected_hmm']['pooled_top1']:.2%} |",'',
        '![Decoder selection and student validation](validation_comparison.png)','',
        'Training OOF values were used for hyperparameter selection and are not an independent post-selection accuracy estimate. Student testing was loaded only after the selected configuration was written; it did not rank trials. Reference labels are derived from monotone route alignment, with the existing 50 m validity threshold.','',
        '## Frozen settings','',
        '```json',json.dumps(selected['decoder_parameters'],indent=2),'```','',
        'The search screened 25 configurations on ten training trips (two per fold, seed 42), then checked the best three and the original decoder on all 78 eligible training trips. The final choice maximized mean-fold Top-1, then mean-trip Top-1; exact ties used trial order. The original folds, reference validity and excluded trips were preserved.','',
        'A small backwards projection within the chosen jitter tolerance is treated as a stationary observation on the same directed arc. It does not create a reverse road edge. Larger backwards motion still needs a directed return path. The optional log-odds emission score is an emission potential, not a fitted probability calibration model.','',
        f"Frozen Random Forest SHA-256: `{selected['model_sha256']}`.",'',
        '## Backend use','',
        'Set `PATHSNAP_DECODER_CONFIG` to the completed `decoder_configuration.json` and restart the backend. The classifier hash is checked before loading the decoder settings. The two-client HTTP request/response contract stays the same; see `docs/CLIENT_API.md`.','']
    if evaluator_run:
        evaluator_run=Path(evaluator_run)
        evaluated=json.loads((evaluator_run/'decoder_configuration.json').read_text())
        if evaluated != selected:
            raise ValueError('Evaluator run did not use the frozen training-selected configuration')
        metrics=json.loads((evaluator_run/'metrics.json').read_text())
        shutil.copy2(evaluator_run/'metrics.json',archive/'repeat_evaluator_metrics.json')
        shutil.copy2(evaluator_run/'per_trip_metrics.csv',archive/'repeat_evaluator_per_trip.csv')
        shutil.copy2(evaluator_run/'run.json',archive/'repeat_evaluator_run.json')
        lines += ['## Repeat evaluator comparison','',
            f"Frozen Random Forest: **{metrics['top1_accuracy']:.2%}**. Training-selected HMM: **{metrics['hmm_top1_accuracy']:.2%}**.",
            f"Both use {metrics['known_observations']:,} valid derived references out of {metrics['observations']:,} observations; {metrics['unresolved_observations']} references are unresolved.",'',
            'This evaluator set had already been examined before this optimization task. The repeated result is a development benchmark, not a fresh untouched final estimate. It was computed after the configuration was frozen and was not used to choose parameters. The original 75.41% HMM benchmark is preserved in `outputs/experiments/evaluator_test_rf_hmm_001/`.','']
    (archive/'README.md').write_text('\n'.join(lines))
    print(archive/'README.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tuning-dir',required=True);p.add_argument('--archive',required=True)
    p.add_argument('--evaluator-run')
    a=p.parse_args();report(a.tuning_dir,a.archive,a.evaluator_run)
