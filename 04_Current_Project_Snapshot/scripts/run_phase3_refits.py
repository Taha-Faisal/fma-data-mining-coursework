"""Refit classical and MLP experiments on the reviewed frozen cohort.

AI-generated code (OpenAI Codex). Full initiating user prompt:
"Continue on with the next phase."
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os

PROJECT=Path(__file__).resolve().parents[1]
os.chdir(PROJECT)
scratch=PROJECT/".work_tmp/phase3"
scratch.mkdir(parents=True,exist_ok=True)
os.environ["MPLCONFIGDIR"]=str(scratch)
os.environ["OMP_NUM_THREADS"]="4"
os.environ["OPENBLAS_NUM_THREADS"]="4"
from run_task3 import run as run_classical
from run_task4 import run as run_mlp

def main():
    out=PROJECT/"results/phase3"
    out.mkdir(parents=True,exist_ok=True)
    pointer=json.loads((PROJECT/"data/processed/task2/LATEST_FROZEN.json").read_text())
    bundle=PROJECT/"data/processed/task2"/pointer["bundle"]
    receipt=dict(status="running",stage="classical",started_utc=datetime.now(timezone.utc).isoformat(),cohort_id=pointer["candidate_id"],test_scored=False)
    def save():
        (out/"refits.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    save()
    try:
        before=set((PROJECT/"results/task3").glob("*/run.json"))
        run_classical(bundle,PROJECT/"data/raw/fma_metadata/features.csv",PROJECT/"results/task3")
        created=set((PROJECT/"results/task3").glob("*/run.json"))-before
        assert len(created)==1
        classical=created.pop().parent
        receipt.update(stage="mlp",classical_run=str(classical.relative_to(PROJECT)))
        save()
        before=set((PROJECT/"results/task4").glob("*/run.json"))
        run_mlp(classical,PROJECT/"data/raw/fma_metadata/features.csv",PROJECT/"results/task4")
        created=set((PROJECT/"results/task4").glob("*/run.json"))-before
        assert len(created)==1
        mlp=created.pop().parent
        receipt.update(status="complete",stage="validation_refits_complete",mlp_run=str(mlp.relative_to(PROJECT)),finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        print("PHASE3_REFITS_COMPLETE="+json.dumps(receipt),flush=True)
    except Exception as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        save()
        raise

if __name__=="__main__":main()
