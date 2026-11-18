# FMA listening engagement coursework

This repository accompanies the team's FMA-medium experiment. The models predict the training-defined listening-engagement label from audio. Use the saved comparisons for inspection and the scripts for replication.

## Verify the delivered result

From the repository root run the following with Python 3.11 or later. The verifier uses the standard library, the frozen manifest and the saved track-level scores. It checks artifact hashes, test membership, labels, the two threshold protocols, confusion counts and hit F1 for all seven models.

~~~sh
python verify_results.py
~~~

The recorded experiment used 20,300 training tracks, 2,546 validation tracks and 2,026 held-out tracks. Results and figures are in **04_Current_Project_Snapshot/results/phase4**. EVALUATION_LOCK.json contains the settings selected before test scoring. TEST_RECEIPT.json identifies the scored predictions. The selected CNN weights are in **results/task6_cnn/frozen_71ff5f58c238b27a/cnn.pt**, with learning curves and experiment tables alongside them.

## Recompute all comparison metrics and uncertainty

Work in a separate clone because the analysis script writes its outputs. From the repository root:

~~~sh
cd 04_Current_Project_Snapshot
python -m venv .venv
~~~

Activate the environment (on Windows, run **.venv/Scripts/Activate.ps1**). Install the recorded scientific-Python requirements and run:

~~~sh
python -m pip install -r requirements-lock.txt
python scripts/run_phase4.py analyze
~~~

This recomputes both protocols and the grouped bootstrap from archived scores. It requires neither source audio nor a GPU. The frozen configuration uses a portable path for the review receipt. Its bundle ledger records the corresponding configuration hash. The cohort ID, manifest, lock, selected weights and archived scores retain their recorded fingerprints.

## Source inputs and training

The [FMA project data page](https://github.com/mdeff/fma#data) provides the original metadata, descriptors and medium MP3 archive. From the snapshot directory, **python scripts/fetch_fma_metadata.py** obtains the metadata. The original features.csv SHA-256 is listed in EVALUATION_LOCK.json. Place the official medium ZIP at **01_Original_Files/FMA_Audio/fma_medium_audio.zip** at repository level. Its recorded SHA-256 is 54517eedebd06ef2de6d99a4afaac225150931b8bbd8b02e29da1cbd7c9aefd1.

For a fresh training replication, run **python scripts/verify_cleaning.py --rebuild** from the snapshot directory after obtaining tracks.csv. This reconstructs the team's cleaned CSV and duplicate log from the official metadata and verifies both original file hashes. The 24,896-row output is **data/raw/fma_medium_clean2.csv** (SHA-256 f094a8df9b4eea58b993e99440763da33d4a47fd9c5c9e5f71537208dae6a178). The original data and large derived arrays are kept separately. The lead's organized collection holds the duplicate log, full conversion, all seven original checkpoints and training environments. The kNN checkpoint contains fitted training features and is available there. Its hash and original location are in the evaluation lock.

In a separate working copy, run **python scripts/run_task5_full.py** from the snapshot directory using the recorded conversion packages. Check the resulting per-track array hashes and locations against the frozen manifest before training. **python scripts/run_phase3_refits.py** fits the classical models and MLP. **python scripts/run_phase3_cnn.py** runs the CUDA learning check and training trials. Use **--verify-only** for just the initial checks. See **results/phase3/requirements-cnn-lock.txt**, **CNN_ENVIRONMENT.json** and the selected CNN experiment plan for the recorded software, hardware and hyperparameters. Retain the published test lock and predictions as the original experiment.

The Git history starts with packaging the existing coursework. Earlier contributions should be attributed using the team's contemporaneous records.
