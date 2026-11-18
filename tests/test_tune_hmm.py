import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pathsnap import tune_hmm
from pathsnap.hmm import load_decoder_parameters
from pathsnap.train import file_hash,write_json
import pytest


def test_grid_and_fold_stratified_screen_are_deterministic():
    grid=tune_hmm.settings_grid()
    assert len(grid)==25
    assert len({json.dumps(item,sort_keys=True) for item in grid})==25
    trips=[SimpleNamespace(trip_id=f'{f}-{i}',fold=f) for f in range(5) for i in range(4)]
    a=tune_hmm.screen_subset(trips)
    b=tune_hmm.screen_subset(trips)
    assert [t.trip_id for t in a]==[t.trip_id for t in b]
    assert len(a)==10
    assert all(sum(t.fold==f for t in a)==2 for f in range(5))


def test_student_data_is_loaded_only_after_training_selection_freezes(tmp_path,monkeypatch):
    features=tmp_path/'features';features.mkdir()
    schema=features/'feature_schema.json';schema.write_text('{}')
    hashes={'schema':file_hash(schema)}
    cv=tmp_path/'cv';(cv/'random_forest').mkdir(parents=True)
    (cv/'random_forest/cv_candidate_scores.csv').write_text('synthetic frozen OOF fixture')
    write_json(cv/'run.json',{'input_sha256':hashes,'input_files':{'schema':str(schema)}})
    model=tmp_path/'model';model.mkdir()
    (model/'pipeline.joblib').write_text('not deserialized by tuning')
    write_json(model/'run.json',{'input_sha256':hashes,'model_sha256':file_hash(model/'pipeline.joblib')})
    output=tmp_path/'output'
    loaded=[]

    def load(feature_dir,raw_root,partition,score_path):
        assert partition in {'train','student_test'}
        if partition=='student_test':
            assert (output/'decoder_configuration.json').is_file()
            assert json.loads((output/'run.json').read_text())['status']=='selected_before_student_validation'
        loaded.append(partition)
        return [tune_hmm.DecoderTrip(str(f),f,
            pd.DataFrame({'point_idx':[0,0,1,1],'arc_id':[0,1,0,1],'frac_along':[.2,.8,.3,.7]}),
            np.array([.9,.1,.9,.1]),np.array([[0,1],[1,0]]),np.array([100.,100.]),
            np.array([[20.,0],[30.,0]]),np.array([0.,1.]),np.array([0,0]),np.array([True,True]))
            for f in range(5)]

    monkeypatch.setattr(tune_hmm,'load_trips',load)
    selected,student=tune_hmm.run(features,tmp_path/'raw',cv,model,output)
    assert loaded==['train','student_test']
    assert not selected['evaluator_used_for_selection']
    assert student['selected_hmm']['pooled_top1']==1.


def test_decoder_config_is_bound_to_model_and_records_selection_provenance(tmp_path):
    path=tmp_path/'decoder.json'
    record={'model_sha256':'frozen_model','evaluator_used_for_selection':False,
            'decoder_parameters':{'beta':100.,'same_arc_jitter_m':5.,'emission_mode':'log_odds'}}
    write_json(path,record)
    assert load_decoder_parameters(path,model_sha256='frozen_model') == record['decoder_parameters']
    with pytest.raises(ValueError,match='different model'):
        load_decoder_parameters(path,model_sha256='other_model')
    record['evaluator_used_for_selection']=True
    write_json(path,record)
    with pytest.raises(ValueError,match='evaluator-free'):
        load_decoder_parameters(path,model_sha256='frozen_model')
