"""Bounded validation threshold study, immutable evaluation lock, and held-out scoring.

Stages must run in order: plan, validate, test, analyze. Training is never rerun.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score, confusion_matrix)

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'results/phase4'
BUNDLE = PROJECT / 'data/processed/task2/frozen_audio_e835f28bc62f'
MODELS = ['Always hit', 'CNN', 'Decision Tree', 'MLP', 'Majority', 'Naive Bayes', 'kNN']
DUMMIES = ['Always hit', 'Majority']

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, obj):
    p=Path(p); temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)

def now(): return datetime.now(timezone.utc).isoformat()

def metric(y, score, threshold):
    p=np.asarray(score)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(hit_f1=float(f1_score(y,p,zero_division=0)),
        macro_f1=float(f1_score(y,p,average='macro',zero_division=0)),
        precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p,zero_division=0)),
        roc_auc=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score)),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def checked_manifest():
    ledger=read(BUNDLE/'bundle.json')['files']
    for name in ('manifest.csv','config.json'):
        assert sha(BUNDLE/name)==ledger[name], name
    frame=pd.read_csv(BUNDLE/'manifest.csv').set_index('track_id',verify_integrity=True)
    assert frame.groupby('group_id').set_split.nunique().max()==1
    assert len(frame)==24872
    return frame

def plan():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'EXPERIMENT_PLAN.json').exists():
        print('Existing predeclared plan retained'); return
    phase3=read(PROJECT/'results/phase3/PHASE3_SUMMARY.json')
    classic=PROJECT/phase3['classical_run'];mlp=PROJECT/phase3['mlp_run'];cnn=PROJECT/phase3['cnn_run']
    models={name:dict(path=(classic/(name.replace(' ','_')+'.joblib')).relative_to(PROJECT).as_posix())
            for name in MODELS if name not in ('CNN','MLP')}
    models['MLP']=dict(path=(mlp/'mlp.joblib').relative_to(PROJECT).as_posix())
    models['CNN']=dict(path=(cnn/'cnn.pt').relative_to(PROJECT).as_posix())
    for value in models.values():value['sha256']=sha(PROJECT/value['path'])
    features=read(BUNDLE/'config.json')['inputs']['features']
    source_paths=['scripts/run_phase4.py','scripts/run_phase3_cnn.py','src/fma_project/cnn_data.py',
        'results/phase3/all_validation_predictions.csv',
        'data/processed/task2/frozen_audio_e835f28bc62f/manifest.csv',
        'data/processed/task2/frozen_audio_e835f28bc62f/config.json']
    info=dict(created_utc=now(),cohort_id=phase3['cohort_id'],models=models,
        features=features,columns=read(classic/'run.json')['feature_columns'],
        sources={name:sha(PROJECT/name) for name in source_paths},
        threshold_grid=[round(i*.05,2) for i in range(1,20)],
        selection='Maximum validation hit F1; ties within 1e-12: closest to 0.5, then higher threshold. Model ties: alphabetical.',
        dummy_threshold=.5,primary_metric='hit_f1',retain_default_threshold_comparison=True,
        retrain=False,additional_seed_runs=False,
        scope='Audio-only; metadata used for grouping, target and descriptive error analysis. No new predictive metadata ablation.',
        uncertainty=dict(seed=20261116,resamples=5000,unit='frozen linked group_id',method='paired percentile bootstrap',
            comparisons='Validation-selected model versus every other model; exploratory, no multiple-comparison adjustment; fixed fitted models only'),
        test_protocol='Replay validation from all seven checkpoints, seal choices before test. Score each checkpoint once; reuse saved scores for all analysis. Report fixed 0.5 and locked selected thresholds for all models; never revise from test outcomes.',
        test_scored=False)
    write(OUT/'EXPERIMENT_PLAN.json',info)
    print('Predeclared threshold grid and evaluation protocol saved',flush=True)

def verify_plan(info):
    for name,digest in info['sources'].items():assert sha(PROJECT/name)==digest, name
    for name,details in info['models'].items():assert sha(PROJECT/details['path'])==details['sha256'],name
    assert sha(PROJECT/info['features']['path'])==info['features']['sha256']

def score(info, split):
    verify_plan(info)
    manifest=checked_manifest(); records=manifest[manifest.set_split.eq(split)]
    assert len(records)==({'validation':2546,'test':2026}[split])
    parts=[]
    for chunk in pd.read_csv(PROJECT/info['features']['path'],header=[0,1,2],index_col=0,chunksize=4000):
        parts.append(chunk.loc[chunk.index.isin(records.index)])
    x=pd.concat(parts,verify_integrity=True).loc[records.index]
    x.columns=['audio__'+'__'.join(c) for c in x.columns]
    x=x[info['columns']].replace([np.inf,-np.inf],np.nan)
    frames=[]
    with threadpool_limits(limits=4):
        for name in MODELS:
            if name=='CNN':continue
            pipe=joblib.load(PROJECT/info['models'][name]['path'])
            s=pipe.predict_proba(x)[:,list(pipe.classes_).index(1)]
            frames.append(pd.DataFrame(dict(model=name,track_id=records.index,hit=records.hit.to_numpy(),score=s)))
            print('Scored',split,name,flush=True)
    import torch
    from run_phase3_cnn import make_model, evaluate, loader, seed_everything
    sys.path.insert(0,str(PROJECT/'src'))
    from fma_project.cnn_data import SpectrogramDataset
    seed_everything(42)
    checkpoint=torch.load(PROJECT/info['models']['CNN']['path'],map_location='cpu',weights_only=True)
    assert checkpoint['cohort_id']==info['cohort_id']
    model=make_model(.3).to('cuda',memory_format=torch.channels_last)
    model.load_state_dict(checkpoint['state_dict'])
    data=SpectrogramDataset(PROJECT,split,bundle=BUNDLE)
    batches=loader(data,False,0,torch.Generator().manual_seed(43))
    _,_,ids,y,s=evaluate(model,batches,torch.nn.BCEWithLogitsLoss())
    frames.append(pd.DataFrame(dict(model='CNN',track_id=ids,hit=y,score=s)))
    result=pd.concat(frames,ignore_index=True)
    for name,frame in result.groupby('model'):
        assert frame.track_id.is_unique and set(frame.track_id)==set(records.index)
        assert np.array_equal(frame.hit,records.loc[frame.track_id,'hit'])
        assert np.isfinite(frame.score).all() and frame.score.between(0,1).all()
    print('Scored',split,'CNN',flush=True)
    return result

def validate():
    if (OUT/'EVALUATION_LOCK.json').exists():
        print('Existing sealed choices retained');return
    info=read(OUT/'EXPERIMENT_PLAN.json');verify_plan(info)
    manifest=checked_manifest(); val=manifest[manifest.set_split.eq('validation')]
    saved=pd.read_csv(PROJECT/'results/phase3/all_validation_predictions.csv')
    trials=[];selected={};comparison=[]
    for name in MODELS:
        data=saved[saved.model.eq(name)].set_index('track_id').loc[val.index]
        assert len(data)==len(val) and data.index.is_unique and np.array_equal(data.hit,val.hit)
        grid=[.5] if name in DUMMIES else info['threshold_grid']
        candidates=[dict(model=name,threshold=t,**metric(data.hit,data.score,t)) for t in grid]
        trials.extend(candidates);best=max(r['hit_f1'] for r in candidates)
        tied=[r for r in candidates if abs(r['hit_f1']-best)<=1e-12]
        chosen=sorted(tied,key=lambda r:(abs(r['threshold']-.5),-r['threshold']))[0]
        selected[name]=chosen['threshold']
        comparison.extend([dict(model=name,protocol='fixed_0.5',threshold=.5,**metric(data.hit,data.score,.5)),
                           dict(protocol='validation_selected',**chosen)])
    pd.DataFrame(trials).to_csv(OUT/'threshold_trials.csv',index=False)
    pd.DataFrame(comparison).to_csv(OUT/'validation_comparison.csv',index=False)
    actual=score(info,'validation')
    expected=saved.set_index(['model','track_id']).sort_index()
    replay=actual.set_index(['model','track_id']).sort_index()
    pd.testing.assert_index_equal(expected.index,replay.index)
    np.testing.assert_array_equal(expected.hit,replay.hit)
    np.testing.assert_allclose(expected.score,replay.score,rtol=1e-5,atol=1e-6)
    for name in MODELS:
        a=expected.loc[name];b=replay.loc[name]
        for t in (.5,selected[name]):np.testing.assert_array_equal(a.score>=t,b.score>=t)
    actual.to_csv(OUT/'validation_replay.csv',index=False)
    eligible=[r for r in comparison if r['protocol']=='validation_selected' and r['model'] not in DUMMIES]
    preferred=sorted(eligible,key=lambda r:(-r['hit_f1'],r['model']))[0]['model']
    write(OUT/'VALIDATION_REPLAY.json',dict(status='passed',models=7,tracks_per_model=2546,
        both_protocol_predictions_match=True,max_score_difference=float(np.max(np.abs(expected.score-replay.score))),verified_utc=now()))
    lock=dict(**info,locked_utc=now(),plan_sha256=sha(OUT/'EXPERIMENT_PLAN.json'),thresholds=selected,
        preferred_from_validation=preferred,validation_comparison_sha256=sha(OUT/'validation_comparison.csv'),
        validation_replay_sha256=sha(OUT/'VALIDATION_REPLAY.json'))
    write(OUT/'EVALUATION_LOCK.json',lock)
    write(OUT/'LOCK_SEAL.json',dict(sha256=sha(OUT/'EVALUATION_LOCK.json'),sealed_utc=now(),test_scored=False))
    print('LOCKED',preferred,selected,flush=True)

def test():
    lock=read(OUT/'EVALUATION_LOCK.json')
    assert sha(OUT/'EVALUATION_LOCK.json')==read(OUT/'LOCK_SEAL.json')['sha256']
    assert sha(OUT/'EXPERIMENT_PLAN.json')==lock['plan_sha256']
    assert sha(OUT/'validation_comparison.csv')==lock['validation_comparison_sha256']
    assert sha(OUT/'VALIDATION_REPLAY.json')==lock['validation_replay_sha256']
    if (OUT/'TEST_RECEIPT.json').exists():
        receipt=read(OUT/'TEST_RECEIPT.json')
        assert sha(OUT/'test_predictions.csv')==receipt['predictions_sha256']
        print('Existing held-out predictions retained; no rescoring');return
    if (OUT/'TEST_STARTED.json').exists():
        raise RuntimeError('A test attempt already began. Inspect it before any restart; do not silently rescore.')
    verify_plan(lock)
    write(OUT/'TEST_STARTED.json',dict(started_utc=now(),lock_sha256=sha(OUT/'EVALUATION_LOCK.json')))
    predictions=score(lock,'test')
    predictions.to_csv(OUT/'test_predictions.csv',index=False)
    write(OUT/'TEST_RECEIPT.json',dict(status='complete',completed_utc=now(),models=7,tracks_per_model=2026,
        prediction_rows=len(predictions),predictions_sha256=sha(OUT/'test_predictions.csv'),
        lock_sha256=sha(OUT/'EVALUATION_LOCK.json'),test_scored=True))
    print('Held-out scores saved; subsequent analysis reuses these files',flush=True)

def analyze():
    lock=read(OUT/'EVALUATION_LOCK.json');receipt=read(OUT/'TEST_RECEIPT.json')
    assert sha(OUT/'test_predictions.csv')==receipt['predictions_sha256']
    assert sha(OUT/'EVALUATION_LOCK.json')==receipt['lock_sha256']
    manifest=checked_manifest();test_rows=manifest[manifest.set_split.eq('test')]
    pred=pd.read_csv(OUT/'test_predictions.csv');rows=[];counts=[]
    groups=np.sort(test_rows.group_id.unique())
    for name in MODELS:
        data=pred[pred.model.eq(name)].set_index('track_id').loc[test_rows.index]
        assert len(data)==2026 and data.index.is_unique and np.array_equal(data.hit,test_rows.hit)
        assert np.isfinite(data.score).all() and data.score.between(0,1).all()
        for protocol,t in [('fixed_0.5',.5),('validation_selected',lock['thresholds'][name])]:
            rows.append(dict(model=name,protocol=protocol,threshold=t,**metric(data.hit,data.score,t)))
        y=data.hit.to_numpy();p=data.score.to_numpy()>=lock['thresholds'][name]
        counts.append(pd.DataFrame(dict(group_id=test_rows.group_id.to_numpy(),tp=((y==1)&p).astype(int),
            fp=((y==0)&p).astype(int),fn=((y==1)&~p).astype(int))).groupby('group_id')[['tp','fp','fn']].sum().reindex(groups).to_numpy())
    pd.DataFrame(rows).to_csv(OUT/'test_comparison.csv',index=False)
    counts=np.stack(counts);rng=np.random.default_rng(lock['uncertainty']['seed']);draws=[]
    for _ in range(lock['uncertainty']['resamples']):
        c=counts[:,rng.integers(0,len(groups),len(groups)),:].sum(axis=1);den=2*c[:,0]+c[:,1]+c[:,2]
        draws.append(np.divide(2*c[:,0],den,out=np.zeros(len(MODELS)),where=den!=0))
    draws=np.asarray(draws);preferred=lock['preferred_from_validation'];index=MODELS.index(preferred)
    uncertainty=dict(protocol=lock['uncertainty'],groups=len(groups),largest_group=int(test_rows.groupby('group_id').size().max()),
        intervals={name:np.quantile(draws[:,i],[.025,.975]).tolist() for i,name in enumerate(MODELS)},
        preferred_minus_other={name:np.quantile(draws[:,index]-draws[:,i],[.025,.975]).tolist() for i,name in enumerate(MODELS) if name!=preferred},
        limitation='Conditional on fitted models and observed group distribution; excludes training variation. Exploratory multiple comparisons, not adjusted significance tests.')
    write(OUT/'TEST_UNCERTAINTY.json',uncertainty)
    write(OUT/'PHASE4_SUMMARY.json',dict(status='complete',cohort_id=lock['cohort_id'],test_scored=True,
        training_tracks=20300,validation_tracks=2546,test_tracks=2026,preferred_from_validation=preferred,
        thresholds=lock['thresholds'],selected_test_results=[r for r in rows if r['protocol']=='validation_selected'],
        completed_utc=now(),scope='Evaluation complete; assessed report, viva, staff access and student evaluation remain separate.'))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['plan','validate','test','analyze'])
    args=parser.parse_args();globals()[args.stage]()
