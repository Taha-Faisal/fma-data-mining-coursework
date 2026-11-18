"""Generate final-cohort development evidence from already saved predictions.

AI-generated code (OpenAI Codex). Full user prompt:
"Could you implement the next changes that you have recommended?"
No training, threshold selection or model scoring occurs here.
"""
from pathlib import Path
import json
import hashlib
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.tree import plot_tree, export_text

P=Path(__file__).resolve().parents[1];OUT=P/'results/phase4';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
lock=read(OUT/'EVALUATION_LOCK.json');summary=read(OUT/'PHASE4_SUMMARY.json')
assert sha(OUT/'test_predictions.csv')==read(OUT/'TEST_RECEIPT.json')['predictions_sha256']
assert sha(OUT/'EVALUATION_LOCK.json')==read(OUT/'LOCK_SEAL.json')['sha256']
manifest=pd.read_csv(P/'data/processed/task2/frozen_audio_e835f28bc62f/manifest.csv').set_index('track_id')
meta=pd.read_csv(P/'data/raw/fma_medium_clean2.csv').set_index('track_id').loc[manifest.index]
assert meta.index.is_unique and manifest.index.equals(meta.index)
cohort=manifest.join(meta[['track_genre_top','track_duration','track_date_created','album_date_released','track_language_code','track_tags']])
cohort['created_year']=pd.to_datetime(cohort.track_date_created,errors='coerce').dt.year.astype('Int64').astype('string').fillna('Unknown')
cohort['track_genre_top']=cohort.track_genre_top.fillna('Unknown')
split=cohort.groupby('set_split').agg(tracks=('hit','size'),hits=('hit','sum'),groups=('group_id','nunique'),artists=('artist_id','nunique'))
split['hit_fraction']=split.hits/split.tracks;split.to_csv(OUT/'cohort_split_summary.csv')
pd.crosstab(cohort.track_genre_top,cohort.set_split,margins=True).to_csv(OUT/'cohort_genres.csv')
cohort.groupby('set_split').track_listens.describe().to_csv(OUT/'listens_summary.csv')
cohort[['track_duration','track_date_created','album_date_released','track_language_code','track_tags']].isna().mean().rename('missing_fraction').to_csv(OUT/'metadata_missingness.csv')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
ordered=split.reindex(['training','validation','test'])
axes[0].bar(ordered.index,ordered.tracks-ordered.hits,label='Non-hit',color='#345e78')
axes[0].bar(ordered.index,ordered.hits,bottom=ordered.tracks-ordered.hits,label='Hit',color='#e79944')
axes[0].set(title='Frozen cohort: 24,872 tracks',ylabel='Tracks');axes[0].legend()
train=cohort[cohort.set_split.eq('training')]
axes[1].hist(np.log10(train.track_listens),bins=45,color='#345e78')
axes[1].axvline(np.log10(2179.25),color='#be4b38',label='Training quartile: 2,179.25')
axes[1].set(title='Training listens distribution',xlabel='log10(recorded listens)',ylabel='Tracks');axes[1].legend(fontsize=8)
fig.savefig(FIG/'cohort_overview.png');plt.close(fig)

validation=pd.read_csv(P/'results/phase3/all_validation_predictions.csv')
test=pd.read_csv(OUT/'test_predictions.csv')
tables={'validation':pd.read_csv(OUT/'validation_comparison.csv'),'test':pd.read_csv(OUT/'test_comparison.csv')}
subgroups=[];error_examples=[]
for name,predictions in [('validation',validation),('test',test)]:
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for model,data in predictions.groupby('model'):
        if model in ['Always hit','Majority']:continue
        pr,rec,_=precision_recall_curve(data.hit,data.score)
        ap=tables[name].query('model == @model and protocol == "validation_selected"').iloc[0].average_precision
        axes[0].plot(rec,pr,label=f'{model}: AP {ap:.3f}',linewidth=1.5)
    prevalence=float(predictions[predictions.model.eq('CNN')].hit.mean())
    axes[0].axhline(prevalence,color='grey',linestyle='--',label=f'Prevalence {prevalence:.3f}')
    axes[0].set(title=f'{name.title()} precision-recall',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));axes[0].legend(fontsize=8)
    result=tables[name].query('protocol == "validation_selected"').set_index('model').loc[lock['preferred_from_validation']]
    matrix=np.array([[result.tn,result.fp],[result.fn,result.tp]],dtype=int)
    axes[1].imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):axes[1].text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=19,color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    axes[1].set(xticks=[0,1],xticklabels=['Non-hit','Hit'],yticks=[0,1],yticklabels=['Non-hit','Hit'],xlabel='Predicted',ylabel='Actual',title=f'{lock["preferred_from_validation"]} on {name}\nLocked threshold {result.threshold:g}')
    fig.savefig(FIG/f'{name}_pr_and_confusion.png');plt.close(fig)
    for model,part in predictions.groupby('model'):
        joined=part.set_index('track_id').join(cohort[['group_id','track_genre_top','created_year']])
        joined['predicted']=joined.score.ge(lock['thresholds'][model])
        for dimension in ['track_genre_top','created_year']:
            for value,g in joined.groupby(dimension):
                subgroups.append(dict(split=name,model=model,dimension=dimension,subgroup=str(value),tracks=len(g),groups=g.group_id.nunique(),hits=int(g.hit.sum()),
                    hit_f1=f1_score(g.hit,g.predicted,zero_division=0),precision=precision_score(g.hit,g.predicted,zero_division=0),recall=recall_score(g.hit,g.predicted,zero_division=0),
                    false_positives=int(((g.hit==0)&g.predicted).sum()),false_negatives=int(((g.hit==1)&~g.predicted).sum()),
                    small_group=len(g)<30 or g.group_id.nunique()<5))
        if model==lock['preferred_from_validation']:
            for kind,mask in [('false_positive',(joined.hit==0)&joined.predicted),('false_negative',(joined.hit==1)&~joined.predicted)]:
                # Deterministic examples, not cherry-picked by confidence or metadata.
                examples=joined.loc[mask].sort_index().head(10).reset_index()
                error_examples.extend(examples.assign(split=name,error=kind).to_dict('records'))
pd.DataFrame(subgroups).to_csv(OUT/'subgroup_errors.csv',index=False)
pd.DataFrame(error_examples).to_csv(OUT/'error_examples.csv',index=False)

fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for axis,(name,table) in zip(axes,tables.items()):
    pivot=table.pivot(index='model',columns='protocol',values='hit_f1').sort_values('validation_selected')
    pos=np.arange(len(pivot));axis.barh(pos-.17,pivot['fixed_0.5'],height=.32,label='Fixed 0.5',color='#aac1ce')
    axis.barh(pos+.17,pivot['validation_selected'],height=.32,label='Locked threshold',color='#345e78')
    axis.set(yticks=pos,yticklabels=pivot.index,xlim=(0,.65),xlabel='Hit F1',title=name.title())
    axis.legend(fontsize=8,loc='lower right')
fig.savefig(FIG/'threshold_comparison.png');plt.close(fig)

tree=joblib.load(P/lock['models']['Decision Tree']['path'])[-1]
fig,axis=plt.subplots(figsize=(19,8),layout='constrained')
plot_tree(tree,max_depth=2,feature_names=lock['columns'],class_names=['Non-hit','Hit'],filled=True,rounded=True,fontsize=7,ax=axis)
axis.set_title('Selected Decision Tree: first three levels\nNode colours show training majority; final hit decisions use locked probability threshold 0.05.')
fig.savefig(FIG/'decision_tree_top_levels.png');plt.close(fig)
(OUT/'decision_tree_rules.txt').write_text(export_text(tree,feature_names=lock['columns'],max_depth=3),encoding='utf-8')

clusters=sorted((P/'results/task6_clustering').glob('frozen_*/run.json'))
clustering=next(p.parent for p in reversed(clusters) if read(p)['manifest_id']==lock['cohort_id'])
scores=pd.read_csv(clustering/'scores.csv');selected=scores[scores.converged].sort_values('validation_silhouette',ascending=False).groupby('model').first()
write(OUT/'CLUSTERING_RECEIPT.json',dict(folder=clustering.relative_to(P).as_posix(),cohort_id=lock['cohort_id'],training_tracks=20300,validation_sample=2000,
    selected={name:dict(k=int(row.k),silhouette=float(row.validation_silhouette)) for name,row in selected.iterrows()},
    run_sha256=sha(clustering/'run.json'),scores_sha256=sha(clustering/'scores.csv')))
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for ax,family in zip(axes,['k-means','GMM']):
    counts=pd.read_csv(clustering/f'{family}_cluster_genres.csv',index_col=0).drop(index='All',columns='All')
    shares=counts.div(counts.sum(axis=1),axis=0)
    im=ax.imshow(shares.to_numpy(),vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set(xticks=np.arange(len(shares.columns)),xticklabels=shares.columns,yticks=np.arange(len(shares)),yticklabels=shares.index,ylabel='Cluster',title=f'{family}: within-cluster genre shares')
    ax.tick_params(axis='x',rotation=75,labelsize=8)
fig.colorbar(im,ax=axes,label='Fraction of cluster');fig.savefig(FIG/'cluster_genres.png');plt.close(fig)

uncertainty=read(OUT/'TEST_UNCERTAINTY.json')
final=tables['test'].query('protocol == "validation_selected"').sort_values('hit_f1',ascending=False)
val=tables['validation'].query('protocol == "validation_selected"').set_index('model')
lines=['# Phase four: locked evaluation and final-cohort analysis','',
    f"Validation selected **{lock['preferred_from_validation']}** before test scoring. All seven models were evaluated on the same **2,026 held-out tracks** with unchanged training weights and locked thresholds. Earlier default-threshold results are preserved. Test data are now evaluated and must not be called untouched in future work.",'',
    '| Model | Threshold | Validation F1 | Test F1 | Test precision | Test recall | Test AP |',
    '|---|---:|---:|---:|---:|---:|---:|']
for r in final.itertuples():lines.append(f'| {r.model} | {r.threshold:g} | {val.loc[r.model,"hit_f1"]:.4f} | {r.hit_f1:.4f} | {r.precision:.4f} | {r.recall:.4f} | {r.average_precision:.4f} |')
preferred=final.set_index('model').loc[lock['preferred_from_validation']]
lines+=['','## Interpretation',
    f"The preselected {lock['preferred_from_validation']} has {int(preferred.tp)} true hits, {int(preferred.fp)} false alarms and {int(preferred.fn)} misses on test. The table reports every model, irrespective of its final rank. The target is historical FMA engagement, not future chart success or causation.",
    '','The bounded threshold study used 19 values from 0.05 to 0.95. Ties favoured proximity to 0.5, then the higher threshold. The tree and MLP selected the lower grid boundary; the search was not expanded after seeing results. Threshold tuning does not change ranking scores (ROC AUC/AP). Validation scores are selection-biased. Fixed 0.5 and selected-threshold test tables use the same saved test scores and are reported as predeclared comparisons, not alternatives from which to select a winner.',
    '',f"Paired group bootstrap used {uncertainty['groups']} test groups, 5,000 resamples and seed 20261116. Largest group: {uncertainty['largest_group']} tracks. Intervals condition on the fitted models and do not capture training randomness or justify claims from multiple unadjusted comparisons.",'',
    '| Preselected model minus comparator | 95% percentile interval for test F1 difference |','|---|---:|']
for name,interval in uncertainty['preferred_minus_other'].items():lines.append(f'| {name} | [{interval[0]:.4f}, {interval[1]:.4f}] |')
lines+=['','## Final-cohort analysis',
    'The frozen sample contains 24,872 tracks. Descriptor clustering was refitted on its 20,300 training tracks using training-fitted imputation, scaling and 30-component PCA; model choice used the same 2,000 sampled validation tracks. Four clusters were selected for both families.',
    f"k-means silhouette: {selected.loc['k-means','validation_silhouette']:.4f}; GMM silhouette: {selected.loc['GMM','validation_silhouette']:.4f}. These weak values do not support sharply separated natural musical categories. Genre tables describe clusters after fitting; genre was not used as input.",
    '','The supervised study is explicitly audio-only. Metadata supplies labels, grouping, provenance and descriptive error analysis. The earlier pitch ambition to compare predictive metadata is deferred. Listens, favourites, interest, names and IDs are not added as predictors. This keeps the implemented question precise while retaining metadata-based error interpretation.',
    '','`subgroup_errors.csv` reports genre and metadata-creation-year errors with track/group counts. Groups with fewer than 30 tracks or five linked groups are flagged as small. Creation year is an exposure-related descriptor, not a verified exposure duration. Subgroup findings are descriptive and were not used to retune the models. `error_examples.csv` selects the first ten IDs for each error type, rather than favourable examples.',
    '','The saved tree illustration shows only the first three levels of the existing fitted tree. A practical use is an inspectable audio-based triage rule or comparison baseline; it is not a causal recipe for creating a hit. Node colours show training-majority classes, while deployed classification follows the locked probability threshold. The existing MLP feature analysis remains valid on the frozen cohort: the top-100 ANOVA trial was weaker than all features, and family permutation importance is associative and affected by descriptor correlations.',
    '','## Reproduction and verification',
    'All seven checkpoints were reloaded and replayed on all validation tracks before the evaluation lock was sealed. Scores matched within tolerance and binary decisions matched exactly at both thresholds. Checkpoint, feature-table, manifest and source-code hashes were verified. One complete set of test scores is saved; analysis and plots read those saved scores without rerunning model inference. See EXPERIMENT_PLAN.json, EVALUATION_LOCK.json, LOCK_SEAL.json, VALIDATION_REPLAY.json and TEST_RECEIPT.json.',
    '','No additional stochastic training seeds, architecture search or predictive metadata ablation were run. No exhaustive acoustic near-duplicate search is claimed. Cross-machine reproduction and student independent evaluation remain separate from successful local replay.',
    '','## Evidence for the team',
    '- `validation_comparison.csv`, `threshold_trials.csv`: bounded validation study and retained default protocol.',
    '- `test_comparison.csv`, `test_predictions.csv`, `TEST_UNCERTAINTY.json`: final evaluation and uncertainty.',
    '- `cohort_split_summary.csv`, `cohort_genres.csv`, `metadata_missingness.csv`: current descriptive tables.',
    '- `figures/`: cohort overview, threshold comparison, validation/test PR and confusion plots, tree and cluster genres.',
    '- `subgroup_errors.csv`, `error_examples.csv`, `CLUSTERING_RECEIPT.json`: interpretation evidence.',
    '','The assessed six-page PDF, contribution appendix, final presentation, repository staff access, peer assessment and independent student evaluation are not completed by this development report. See SUBMISSION_HANDOFF.md for the remaining responsibilities.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Saved report, descriptive tables and figures',flush=True)
