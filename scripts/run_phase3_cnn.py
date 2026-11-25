"""Train the planned spectrogram CNN using frozen train/validation data only.

API references: https://docs.pytorch.org/docs/2.11/notes/amp_examples.html
https://docs.pytorch.org/docs/2.11/generated/torch.nn.BCEWithLogitsLoss.html
https://docs.pytorch.org/docs/2.11/notes/randomness.html
"""
from __future__ import annotations
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))
from fma_project.cnn_data import SpectrogramDataset


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def write_csv(path,rows):
    if not rows:return
    temp=Path(path).with_suffix(".csv.tmp")
    with temp.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def checkpoint(path,payload):
    temp=path.with_suffix(".pt.tmp");torch.save(payload,temp);temp.replace(path)


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    torch.set_num_threads(1)
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)


def make_model(dropout=.3):
    return nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                         nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Dropout(dropout),nn.Linear(64,1))


def metrics(truth,scores):
    truth=np.asarray(truth,dtype=np.int64).ravel();scores=np.asarray(scores,dtype=np.float64).ravel()
    predicted=scores>=.5
    tn,fp,fn,tp=confusion_matrix(truth,predicted,labels=[0,1]).ravel()
    return dict(hit_f1=float(f1_score(truth,predicted,zero_division=0)),
                macro_f1=float(f1_score(truth,predicted,average="macro",zero_division=0)),
                precision=float(precision_score(truth,predicted,zero_division=0)),
                recall=float(recall_score(truth,predicted,zero_division=0)),
                roc_auc=float(roc_auc_score(truth,scores)),average_precision=float(average_precision_score(truth,scores)),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def loader(dataset,shuffle,workers,generator):
    options=dict(batch_size=16,shuffle=shuffle,num_workers=workers,pin_memory=True,
                 generator=generator,worker_init_fn=seed_worker)
    if workers:options.update(persistent_workers=True,prefetch_factor=2)
    return DataLoader(dataset,**options)


def evaluate(model,batches,loss_function):
    model.eval();truth=[];scores=[];ids=[];total_loss=0.;count=0
    with torch.inference_mode():
        for x,y,tid in batches:
            x=x.to("cuda",non_blocking=True,memory_format=torch.channels_last)
            y=y.to("cuda",non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                logits=model(x);loss=loss_function(logits,y)
            total_loss+=float(loss)*len(y);count+=len(y)
            scores.extend(torch.sigmoid(logits.float()).cpu().numpy().ravel().tolist())
            truth.extend(y.cpu().numpy().ravel().astype(int).tolist());ids.extend(tid.numpy().tolist())
    assert count==len(batches.dataset) and len(set(ids))==count
    return metrics(truth,scores),total_loss/count,np.asarray(ids),np.asarray(truth),np.asarray(scores)


def tiny_learning_check(training,folder):
    seed_everything(42)
    rng=np.random.default_rng(42)
    index=[]
    for label in (0,1):
        choices=[i for i,row in enumerate(training.rows) if int(row["hit"])==label]
        index.extend(rng.choice(choices,size=4,replace=False).tolist())
    x,y,ids=training.batch(index)
    x=torch.from_numpy(x).to("cuda",memory_format=torch.channels_last);y=torch.from_numpy(y).to("cuda")
    model=make_model(0).to("cuda",memory_format=torch.channels_last)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    criterion=nn.BCEWithLogitsLoss()
    model.eval()
    with torch.no_grad():initial=float(criterion(model(x),y))
    history=[];passed=False;gradient_verified=False
    for step in range(1,801):
        model.train();optimizer.zero_grad(set_to_none=True);loss=criterion(model(x),y)
        if not torch.isfinite(loss):raise ValueError("Nonfinite tiny-subset loss")
        loss.backward()
        if step==1:
            gradient_verified=any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.parameters())
        optimizer.step()
        if step%25==0:
            model.eval()
            with torch.no_grad():
                logits=model(x);value=float(criterion(logits,y));accuracy=float(((logits>=0)==(y>=.5)).float().mean())
            history.append(dict(step=step,loss=value,accuracy=accuracy))
            print("TINY_CHECK",json.dumps(history[-1]),flush=True)
            if value<initial*.5 and accuracy==1.0:
                passed=True;break
    result=dict(passed=passed,gradient_verified=gradient_verified,tracks=ids.tolist(),training_only=True,
                initial_loss=initial,final_loss=value,accuracy=accuracy,steps=step,
                dropout=0.,learning_rate=.003,note="Debugging check only. Weights discarded before full experiments.")
    write_json(folder/"tiny_learning_check.json",result);write_csv(folder/"tiny_learning_history.csv",history)
    del model,optimizer,x,y;gc.collect();torch.cuda.empty_cache()
    if not passed or not gradient_verified:raise ValueError("Tiny-subset learning check did not pass")
    return result


def train_trial(settings,training,validation,folder,config,workers,receipt_path,receipt):
    trial=folder/settings["name"];trial.mkdir(exist_ok=True)
    if (trial/"summary.json").exists():
        previous=json.loads((trial/"summary.json").read_text())
        assert previous["cohort_id"]==training.cohort_id and sha(trial/"best.pt")==previous["checkpoint_sha256"]
        print("REUSE_COMPLETED_TRIAL",settings["name"],flush=True)
        return previous
    seed_everything(42)
    train_generator=torch.Generator().manual_seed(42)
    val_generator=torch.Generator().manual_seed(43)
    model=make_model(.3).to("cuda",memory_format=torch.channels_last)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings["learning_rate"])
    scaler=torch.amp.GradScaler("cuda")
    weight=torch.tensor([settings["pos_weight"]],device="cuda") if settings["pos_weight"]!=1 else None
    loss_function=nn.BCEWithLogitsLoss(pos_weight=weight)
    start_epoch=1;best_f1=-1.;best_epoch=0;history=[]
    if (trial/"last.pt").exists():
        previous=torch.load(trial/"last.pt",map_location="cpu",weights_only=True)
        assert previous["cohort_id"]==training.cohort_id
        model.load_state_dict(previous["state_dict"]);optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"]);train_generator.set_state(previous["loader_rng"])
        torch.set_rng_state(previous["torch_rng"]);torch.cuda.set_rng_state_all(previous["cuda_rng"])
        start_epoch=previous["epoch"]+1;best_f1=previous["best_f1"];best_epoch=previous["best_epoch"]
        with (trial/"history.csv").open(encoding="utf-8",newline="") as f:
            history=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        print("RESUME_TRIAL",settings["name"],start_epoch,flush=True)
    train_batches=loader(training,True,workers,train_generator)
    val_batches=loader(validation,False,workers,val_generator)
    started=time.monotonic()
    for epoch in range(start_epoch,31):
        if best_epoch and epoch-best_epoch>5:break
        model.train();total_loss=0.;seen=0;epoch_started=time.monotonic()
        for batch,(x,y,_) in enumerate(train_batches,1):
            x=x.to("cuda",non_blocking=True,memory_format=torch.channels_last);y=y.to("cuda",non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                logits=model(x);loss=loss_function(logits,y)
            if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            total_loss+=float(loss.detach())*len(y);seen+=len(y)
            if batch%200==0:
                receipt.update(stage="cnn_training",trial=settings["name"],epoch=epoch,batch=batch,
                               batches_per_epoch=len(train_batches),tracks_processed_in_epoch=seen,
                               elapsed_epoch_seconds=time.monotonic()-epoch_started)
                write_json(receipt_path,receipt)
                write_json(folder/"progress.json",receipt)
                print("BATCH",json.dumps({k:receipt[k] for k in ["trial","epoch","batch","batches_per_epoch","elapsed_epoch_seconds"]}),flush=True)
        result,val_loss,ids,truth,scores=evaluate(model,val_batches,loss_function)
        row=dict(epoch=epoch,train_objective_loss=total_loss/seen,validation_objective_loss=val_loss,
                 validation_hit_f1=result["hit_f1"],validation_average_precision=result["average_precision"],
                 epoch_seconds=time.monotonic()-epoch_started)
        history.append(row);write_csv(trial/"history.csv",history)
        if result["hit_f1"]>best_f1:
            best_f1=result["hit_f1"];best_epoch=epoch
            checkpoint(trial/"best.pt",dict(state_dict=model.state_dict(),epoch=epoch,settings=settings,
                                          cohort_id=training.cohort_id,threshold=.5,metrics=result))
            predictions=[dict(track_id=int(tid),hit=int(y),score=float(s),prediction=int(s>=.5)) for tid,y,s in zip(ids,truth,scores)]
            write_csv(trial/"validation_predictions.csv",predictions)
        checkpoint(trial/"last.pt",dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                                      epoch=epoch,best_epoch=best_epoch,best_f1=best_f1,cohort_id=training.cohort_id,
                                      loader_rng=train_generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        receipt.update(stage="cnn_validation",trial=settings["name"],epoch=epoch,validation_hit_f1=result["hit_f1"],
                       best_validation_hit_f1=best_f1,best_epoch=best_epoch)
        write_json(receipt_path,receipt);write_json(folder/"progress.json",receipt)
        print("EPOCH",settings["name"],json.dumps(row),"BEST",best_epoch,round(best_f1,6),flush=True)
        if epoch-best_epoch>=5:break
    best=torch.load(trial/"best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(best["state_dict"])
    result,_,ids,truth,scores=evaluate(model,val_batches,loss_function)
    with (trial/"validation_predictions.csv").open(encoding="utf-8",newline="") as f:
        saved=list(csv.DictReader(f))
    assert ids.tolist()==[int(row["track_id"]) for row in saved]
    np.testing.assert_allclose(scores,[float(row["score"]) for row in saved],rtol=1e-5,atol=1e-6)
    summary=dict(trial=settings["name"],learning_rate=settings["learning_rate"],pos_weight=settings["pos_weight"],
                 best_epoch=best_epoch,epochs_run=int(history[-1]["epoch"]),cohort_id=training.cohort_id,
                 checkpoint_sha256=sha(trial/"best.pt"),checkpoint_reload_verified=True,validation_tracks=len(validation),
                 elapsed_seconds=time.monotonic()-started,**result)
    write_json(trial/"summary.json",summary)
    del model,optimizer,scaler,train_batches,val_batches;gc.collect();torch.cuda.empty_cache()
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    scratch=PROJECT/".work_tmp/phase3_cnn";scratch.mkdir(parents=True,exist_ok=True)
    os.environ["MPLCONFIGDIR"]=str(scratch)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; full run requires verified GPU")
    pointer=json.loads((PROJECT/"data/processed/task2/LATEST_FROZEN.json").read_text())
    bundle=PROJECT/"data/processed/task2"/pointer["bundle"]
    frozen=json.loads((bundle/"config.json").read_text())
    for name,checksum in json.loads((bundle/"bundle.json").read_text())["files"].items():
        assert sha(bundle/name)==checksum,f"Changed frozen evidence: {name}"
    training=SpectrogramDataset(PROJECT,"training",bundle)
    validation=SpectrogramDataset(PROJECT,"validation",bundle)
    assert len(training)==20300 and len(validation)==2546
    positives=sum(int(row["hit"]) for row in training.rows)
    ratio=(len(training)-positives)/positives
    versions={name:importlib.metadata.version(name) for name in ["torch","numpy","pandas","scikit-learn","scipy","matplotlib"]}
    plan=dict(cohort_id=pointer["candidate_id"],manifest_sha256=sha(bundle/"manifest.csv"),
              script_sha256=sha(Path(__file__)),loader_sha256=sha(PROJECT/"src/fma_project/cnn_data.py"),
              training_tracks=len(training),validation_tracks=len(validation),reserved_test_tracks=2026,
              cutoff=frozen["target"],shape=[1,128,1292],normalization="Fixed (dB+80)/80, clipped to [0,1]",
              architecture="Conv 1->16->32->64, kernel 3 padding 1, ReLU/max-pool each, global average pool, dropout .3, linear 64->1",
              optimizer="Adam",batch_size=16,max_epochs=30,patience=5,seed=42,threshold=.5,
              checkpoint_selection="Highest validation hit F1, first epoch/settings on ties",
              declared_trials=[dict(name="lr001_unweighted",learning_rate=.001,pos_weight=1.),
                               dict(name="lr0003_unweighted",learning_rate=.0003,pos_weight=1.)],
              followup="Train pos_weight=training_nonhits/training_hits at the learning rate selected by validation F1 from the first two trials",
              training_positive_weight=ratio,dropout=.3,augmentation="none",workers=args.workers,
              precision="CUDA automatic mixed precision float16 with gradient scaling",
              deterministic_algorithms=True,channels_last=True,versions=versions,python=platform.python_version(),
              cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
              test_used_for_selection=False,implementation_references=[
                  "https://docs.pytorch.org/docs/2.11/notes/amp_examples.html",
                  "https://docs.pytorch.org/docs/2.11/generated/torch.nn.BCEWithLogitsLoss.html",
                  "https://docs.pytorch.org/docs/2.11/notes/randomness.html"])
    run_id=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:16]
    folder=PROJECT/"results/task6_cnn"/("frozen_"+run_id);folder.mkdir(parents=True,exist_ok=True)
    if (folder/"EXPERIMENT_PLAN.json").exists():assert json.loads((folder/"EXPERIMENT_PLAN.json").read_text())==plan
    write_json(folder/"EXPERIMENT_PLAN.json",plan)
    receipt_path=PROJECT/"results/phase3/cnn_run.json";receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt=dict(status="running",stage="gpu_learning_check",started_utc=datetime.now(timezone.utc).isoformat(),
                 folder=str(folder.relative_to(PROJECT)),cohort_id=training.cohort_id,test_scored=False)
    write_json(receipt_path,receipt)
    print("CNN_RUN_FOLDER="+str(folder),flush=True)
    try:
        if not (folder/"tiny_learning_check.json").exists():tiny=tiny_learning_check(training,folder)
        else:tiny=json.loads((folder/"tiny_learning_check.json").read_text());assert tiny["passed"]
        environment=dict(status="verified",gpu=plan["gpu"],cuda=plan["cuda"],cudnn=plan["cudnn"],versions=versions,
                         cuda_available=True,real_audio_forward_backward_verified=True,tiny_training_check=tiny,
                         test_arrays_loaded=False)
        write_json(PROJECT/"results/phase3/CNN_ENVIRONMENT.json",environment)
        if args.verify_only:
            receipt.update(status="ready",stage="gpu_and_learning_checks_passed");write_json(receipt_path,receipt);return
        completed=[]
        for settings in plan["declared_trials"]:
            completed.append(train_trial(settings,training,validation,folder,plan,args.workers,receipt_path,receipt))
        rate=max(completed,key=lambda r:r["hit_f1"])["learning_rate"]
        weighted=dict(name="selected_lr_weighted",learning_rate=rate,pos_weight=ratio)
        write_json(folder/"WEIGHTED_FOLLOWUP.json",dict(settings=weighted,selection_source="Validation hit F1 from the two predeclared unweighted trials"))
        completed.append(train_trial(weighted,training,validation,folder,plan,args.workers,receipt_path,receipt))
        winner=max(completed,key=lambda row:row["hit_f1"])
        shutil.copy2(folder/winner["trial"]/"best.pt",folder/"cnn.pt")
        shutil.copy2(folder/winner["trial"]/"validation_predictions.csv",folder/"validation_predictions.csv")
        write_csv(folder/"experiments.csv",completed)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
        all_history=[]
        for row in completed:
            with (folder/row["trial"]/"history.csv").open(encoding="utf-8",newline="") as f:
                history=[{k:float(v) for k,v in x.items()} for x in csv.DictReader(f)]
            all_history.extend([dict(trial=row["trial"],**h) for h in history])
            epochs=[h["epoch"] for h in history]
            line,=axes[0].plot(epochs,[h["validation_objective_loss"] for h in history],label=row["trial"])
            axes[0].plot(epochs,[h["train_objective_loss"] for h in history],linestyle="--",color=line.get_color())
            axes[1].plot(epochs,[h["validation_hit_f1"] for h in history],label=row["trial"])
        axes[0].set(xlabel="Epoch",ylabel="BCE objective (weighted in weighted trial)",title="Train (dashed) and validation (solid)")
        axes[1].set(xlabel="Epoch",ylabel="Hit F1 at threshold 0.5",title="Validation checkpoint selection")
        axes[1].legend(fontsize=8);fig.savefig(folder/"learning_curves.png",dpi=160);plt.close(fig)
        write_csv(folder/"learning_curves.csv",all_history)
        summary=dict(status="complete",cohort_id=training.cohort_id,selected_trial=winner["trial"],
                     selected_checkpoint_sha256=sha(folder/"cnn.pt"),validation_tracks=len(validation),
                     training_tracks=len(training),reserved_test_tracks=2026,test_scored=False,
                     checkpoint_reload_verified=all(row["checkpoint_reload_verified"] for row in completed),
                     selected_metrics={key:winner[key] for key in metrics([0,1],[0.,1.])},
                     trials=completed,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder/"summary.json",summary)
        receipt.update(status="complete",stage="cnn_validation_experiments_complete",selected_trial=winner["trial"],
                       validation_hit_f1=winner["hit_f1"],finished_utc=summary["finished_utc"])
        write_json(receipt_path,receipt);write_json(folder/"progress.json",receipt)
        print("CNN_EXPERIMENTS_COMPLETE="+json.dumps(summary),flush=True)
    except Exception as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        write_json(receipt_path,receipt);write_json(folder/"progress.json",receipt)
        raise


if __name__=="__main__":main()
