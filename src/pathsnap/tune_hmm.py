"""Select decoder settings on training OOF scores, then validate on student-test.

Never opens evaluator-test data or uses its measured results as an objective.
"""
import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from . import config, data_io
from .hmm import viterbi_decode
from .network import build_geom
from .train import file_hash, write_json


@dataclass
class DecoderTrip:
    trip_id: str
    fold: int
    candidates: pd.DataFrame
    probabilities: np.ndarray
    arcs: np.ndarray
    lengths: np.ndarray
    xy: np.ndarray
    times: np.ndarray
    reference: np.ndarray
    known: np.ndarray


def load_trips(feature_dir, raw_root, partition, score_path):
    feature_dir = Path(feature_dir)
    points = pd.read_parquet(feature_dir / f'{partition}_points.parquet',
        columns=['trip_id','point_idx','label_arc','label_valid','fold'])
    if partition == 'train':
        points = points.loc[points.fold.ge(0)].copy()
    candidates = pd.read_parquet(feature_dir / f'{partition}_candidates.parquet',
        columns=['trip_id','point_idx','arc_id','frac_along'])
    candidates = candidates.loc[candidates.trip_id.isin(points.trip_id.unique())].copy()
    scores = pd.read_csv(score_path, dtype={'trip_id':'string', 'observation_id':'int64',
        'candidate_id':'int64', 'probability':'float64'})
    if set(scores.trip_id) != set(points.trip_id):
        raise ValueError('Score trip IDs disagree with eligible partition trips')
    if partition == 'train':
        expected = points[['trip_id','fold']].drop_duplicates()
        matched = scores[['trip_id','fold']].drop_duplicates().merge(expected,on='trip_id',suffixes=('_score','_expected'),validate='one_to_one')
        if not matched.fold_score.eq(matched.fold_expected).all():
            raise ValueError('OOF score folds disagree with the original training folds')
    joined = candidates.merge(scores, left_on=['trip_id','point_idx','arc_id'],
        right_on=['trip_id','observation_id','candidate_id'], how='outer', validate='one_to_one',indicator=True)
    if joined._merge.ne('both').any() or not np.isfinite(joined.probability).all():
        raise ValueError('Every candidate needs exactly one aligned frozen score')
    point_groups = dict(tuple(points.groupby('trip_id',sort=True)))
    trips = []
    for trip_id, rows in joined.groupby('trip_id',sort=True):
        trip = data_io.load_trip(str(trip_id),partition,with_route=False,root=raw_root)
        geom = build_geom(trip)
        inventory = point_groups[trip_id].sort_values('point_idx')
        if not np.array_equal(inventory.point_idx.to_numpy(),np.arange(len(trip.track))):
            raise ValueError('Complete point inventory must match raw observation order')
        trips.append(DecoderTrip(str(trip_id),int(inventory.fold.iloc[0]),
            rows[['point_idx','arc_id','frac_along']].reset_index(drop=True),
            rows.probability.to_numpy(),trip.arcs,geom.arc_len,geom.P,geom.t,
            inventory.label_arc.fillna(-1).to_numpy(dtype=int),inventory.label_valid.to_numpy(dtype=bool)))
    return trips


def evaluate_trips(trips, parameters, *, classifier_only=False, save=None):
    rows = []
    started = perf_counter()
    for trip in trips:
        if classifier_only:
            ranked = trip.candidates.assign(probability=trip.probabilities,
                tie=trip.candidates.arc_id.astype(str)).sort_values('tie',kind='stable').sort_values(
                    'probability',ascending=False,kind='stable').drop_duplicates('point_idx')
            predicted = np.full(len(trip.times),-1,dtype=int)
            predicted[ranked.point_idx.to_numpy(dtype=int)] = ranked.arc_id.to_numpy(dtype=int)
            segments = None
        else:
            decoded = viterbi_decode(trip.candidates,trip.arcs,trip.probabilities,
                arc_lengths=trip.lengths,gps_xy=trip.xy,timestamps=trip.times,
                return_details=True,**parameters)
            predicted = np.asarray(decoded.point_arc_ids)
            segments = len(decoded.route_segments)
            for route in decoded.route_segments:
                if any(trip.arcs[a,1] != trip.arcs[b,0] for a,b in zip(route,route[1:])):
                    raise RuntimeError('Decoded route segment is disconnected')
        correct = int(((predicted == trip.reference) & trip.known).sum())
        known = int(trip.known.sum())
        rows.append({'trip_id':trip.trip_id,'fold':trip.fold,'known':known,'correct':correct,
            'top1_accuracy':correct/known if known else np.nan,'segments':segments,
            'observations':len(trip.times),'unresolved':int((~trip.known).sum())})
        if save is not None:
            save.mkdir(parents=True,exist_ok=True)
            pd.DataFrame({'point_idx':np.arange(len(predicted)),'predicted_arc_id':predicted}).to_csv(save/f'{trip.trip_id}.csv',index=False)
    table = pd.DataFrame(rows)
    by_fold = table.groupby('fold')[['correct','known']].sum()
    fold_accuracy = by_fold.correct / by_fold.known
    result = {'mean_fold_top1':float(fold_accuracy.mean()),
        'pooled_top1':float(table.correct.sum()/table.known.sum()),
        'mean_trip_top1':float(table.top1_accuracy.mean()),'correct':int(table.correct.sum()),
        'known':int(table.known.sum()),'trips':len(trips),
        'segments':int(table.segments.sum()) if not classifier_only else None,
        'seconds':perf_counter()-started}
    return result,table


def settings_grid():
    baseline={'beta':50.0,'same_arc_jitter_m':0.0,'emission_mode':'probability'}
    grid=[baseline]
    for beta,jitter,mode in itertools.product([25.0,100.0,500.0,2000.0],[0.0,5.0,15.0],['probability','log_odds']):
        grid.append({'beta':beta,'same_arc_jitter_m':jitter,'emission_mode':mode})
    return grid


def screen_subset(trips,seed=42):
    rng=np.random.default_rng(seed)
    selected=[]
    for fold in range(5):
        eligible=sorted([trip for trip in trips if trip.fold==fold],key=lambda t:t.trip_id)
        selected.extend(eligible[int(i)] for i in rng.choice(len(eligible),size=min(2,len(eligible)),replace=False))
    return selected


def run(feature_dir,raw_root,cv_run,model_dir,output):
    feature_dir,cv_run,model_dir,output=map(Path,[feature_dir,cv_run,model_dir,output])
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a fresh tuning output directory')
    prior=json.loads((cv_run/'run.json').read_text())
    model=json.loads((model_dir/'run.json').read_text())
    if prior['input_sha256'] != model['input_sha256']:
        raise ValueError('OOF experiments and final model must use identical feature inputs')
    for key,path in prior['input_files'].items():
        actual=feature_dir / Path(path).name
        if file_hash(actual) != prior['input_sha256'][key]:
            raise ValueError(f'Changed feature input: {key}')
    artifact_hash=file_hash(model_dir/'pipeline.joblib')
    if artifact_hash != model['model_sha256']:
        raise ValueError('Final model changed')
    output.mkdir(parents=True,exist_ok=True)
    cv_scores=cv_run/'random_forest/cv_candidate_scores.csv'
    manifest={'status':'screening_training_oof','selection_data':'training out-of-fold predictions only',
        'student_test_role':'one post-selection check; not used to rank configurations',
        'evaluator_read':False,'seed':42,'model_sha256':artifact_hash,
        'oof_scores_sha256':file_hash(cv_scores),'grid':settings_grid()}
    write_json(output/'run.json',manifest)
    print('Loading original training OOF predictions and metric road graphs...',flush=True)
    trips=load_trips(feature_dir,raw_root,'train',cv_scores)
    subset=screen_subset(trips)
    manifest['training_trips']=[trip.trip_id for trip in trips]
    manifest['screening_trips']=[trip.trip_id for trip in subset]
    write_json(output/'run.json',manifest)
    rf,rf_table=evaluate_trips(trips,{},classifier_only=True)
    write_json(output/'training_rf.json',rf)
    rf_table.to_csv(output/'training_rf_per_trip.csv',index=False)
    results=[]
    for index,parameters in enumerate(manifest['grid']):
        score,_=evaluate_trips(subset,parameters)
        results.append({'trial':index,**parameters,**score})
        pd.DataFrame(results).to_csv(output/'screening_trials.csv',index=False)
        print(f"Screen {index+1}/{len(manifest['grid'])}: {parameters} -> fold Top-1 {score['mean_fold_top1']:.2%} ({score['seconds']:.1f}s)",flush=True)
    # All original folds take part in screening; finalists run on all 78 trips.
    ranked=sorted(results,key=lambda r:(-r['mean_fold_top1'],-r['mean_trip_top1'],r['trial']))
    finalists=list(dict.fromkeys([0]+[r['trial'] for r in ranked[:3]]))
    manifest['status']='full_training_oof_validation'
    manifest['finalist_trials']=finalists
    write_json(output/'run.json',manifest)
    full=[]
    for trial in finalists:
        parameters=manifest['grid'][trial]
        score,table=evaluate_trips(trips,parameters)
        full.append({'trial':trial,**parameters,**score})
        table.to_csv(output/f'training_trial_{trial}_per_trip.csv',index=False)
        pd.DataFrame(full).to_csv(output/'full_training_trials.csv',index=False)
        print(f"Full training OOF trial {trial}: {score['mean_fold_top1']:.2%}, RF baseline {rf['mean_fold_top1']:.2%}",flush=True)
    best=min(full,key=lambda r:(-r['mean_fold_top1'],-r['mean_trip_top1'],r['trial']))
    selected={'decoder_parameters':manifest['grid'][best['trial']],
        'model_sha256':artifact_hash,'selection_basis':'Best mean-fold Top-1 on all eligible training OOF trips among screened finalists',
        'training_trial':best['trial'],'training_metrics':best,'evaluator_used_for_selection':False}
    write_json(output/'decoder_configuration.json',selected)
    manifest['status']='selected_before_student_validation'
    write_json(output/'run.json',manifest)
    print(f"Frozen selected settings: {selected['decoder_parameters']}",flush=True)
    del trips,subset
    student=load_trips(feature_dir,raw_root,'student_test',model_dir/'student_test_candidate_scores.csv')
    checks={}
    for name,parameters,classifier in [('random_forest',{},True),('original_hmm',manifest['grid'][0],False),('selected_hmm',selected['decoder_parameters'],False)]:
        score,table=evaluate_trips(student,parameters,classifier_only=classifier,
            save=output/'student_predictions'/name)
        checks[name]=score
        table.to_csv(output/f'student_{name}_per_trip.csv',index=False)
        print(f"Student-test {name}: {score['pooled_top1']:.2%}",flush=True)
    write_json(output/'student_validation.json',checks)
    manifest['status']='complete'
    write_json(output/'run.json',manifest)
    return selected,checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir',default='outputs/features')
    parser.add_argument('--raw-root',default=str(config.DATA_DIR))
    parser.add_argument('--cv-run',default='outputs/training/run_001')
    parser.add_argument('--model-dir',default='outputs/training/random_forest_selected_001')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.feature_dir,args.raw_root,args.cv_run,args.model_dir,args.output)


if __name__=='__main__':
    main()
