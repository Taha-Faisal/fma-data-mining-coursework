# What makes a hit? Predicting listening engagement from audio (FMA)

F20DL Data Mining and Machine Learning group coursework, Heriot-Watt University, group UG11.

## Overview

We ask whether a track's audio alone can tell us if it will become a "hit" on the Free Music Archive. A track is labelled a hit when its listen count is in the top quarter of the training split (75th percentile = 2,179.25 listens, so a hit has at least 2,180 listens). Metadata is used only for labels, grouping and error analysis, never as a model input.

The project covers:

- **Data cleaning and exploration**: duplicate removal, missing-value audit, listen distribution and genre balance.
- **Grouped data splits**: tracks by the same artist or album never cross splits (20,300 train / 2,546 validation / 2,026 test, 24,872 tracks in total).
- **Clustering**: k-means and a Gaussian mixture model (EM) on the FMA audio descriptors.
- **Classical models**: Naive Bayes, Decision Tree and kNN on the audio descriptors, against "always hit" and "majority" baselines.
- **Neural networks**: an MLP on the descriptors and a CNN on mel spectrograms.
- **Evaluation**: thresholds were chosen on validation and locked before the test set was scored once. The main metric is F1 for the hit class, with grouped bootstrap intervals.

Test results with the validation-selected thresholds (`results/phase4/test_comparison.csv`):

| Model | Hit F1 |
|---|---|
| CNN (spectrograms) | 0.453 |
| MLP | 0.434 |
| kNN | 0.417 |
| Always hit (baseline) | 0.411 |
| Decision Tree | 0.398 |
| Naive Bayes | 0.395 |
| Majority (baseline) | 0.000 |

## Dataset

**FMA: A Dataset for Music Analysis**, medium subset (Defferrard, Benzi, Vandergheynst and Bresson, ISMIR 2017).

- Source: <https://github.com/mdeff/fma>. Metadata and pre-computed audio features: `fma_metadata.zip`. Audio: `fma_medium.zip` (25,000 30-second MP3 clips, about 22 GB).
- Licence: the metadata and features are released under Creative Commons Attribution 4.0 (CC BY 4.0). Each audio clip keeps the Creative Commons licence its artist chose. The dataset is provided for research, and we used it only for non-commercial academic machine learning work. It holds no personal data beyond public artist names.
- Raw data is not stored in this repository because of its size. `scripts/fetch_fma_metadata.py` downloads and checksums the metadata. Place the audio archive at `data/fma_audio/fma_medium_audio.zip`; its expected size and hashes are in `data/fma_audio/AUDIO_VERIFICATION.json`.

## Repository structure

```text
.
├── README.md
├── requirements.txt            # Loose dependency list
├── requirements-lock.txt       # Exact versions used for the analysis
├── verify_results.py           # Standard-library check of the final test results
├── src/fma_project/            # Reusable code: data audit, grouped splits, spectrogram loader
├── scripts/                    # Pipeline steps, in the order listed below
├── tests/                      # Unit tests on synthetic data
├── data/
│   ├── fma_audio/              # Checksum record for the audio archive (archive itself not committed)
│   ├── processed/task2/        # Frozen cohort: manifest, splits, labels, grouping links, exclusions
│   ├── processed/spectrograms/ # Package lock for the spectrogram conversion (arrays not committed)
│   └── review/                 # Cohort review record
└── results/
    ├── task3/                  # Naive Bayes, Decision Tree, kNN and baselines: models and validation runs
    ├── task4/                  # MLP: model, trials, learning curves, feature-family importance
    ├── task6_clustering/       # k-means and GMM scores and cluster-genre tables
    ├── task6_cnn/              # CNN weights, trials, learning curves
    ├── phase3/                 # Validation comparison of all models on the frozen cohort
    └── phase4/                 # Locked thresholds, single test run, uncertainty, error analysis
        └── figures/            # Figures used in the report
```

### Pipeline

Run each script from the repository root. Steps 1 to 7 need the raw FMA data; steps 8 and 9 work from the files in this repository.

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/fetch_fma_metadata.py` | Downloads and verifies the FMA metadata |
| 2 | `scripts/verify_cleaning.py --rebuild` | Rebuilds the cleaned track list (duplicates and invalid rows removed) and checks its hash |
| 3 | `scripts/run_task1.py` | Audits the cleaned data (missing values, listens, duplicates) |
| 4 | `scripts/run_task2.py` | Builds the grouped train/validation/test manifest and hit labels |
| 5 | `scripts/run_task5_full.py` (uses `run_task5.py`), then `scripts/verify_task5_full.py` | Converts the audio to mel spectrograms and checks every array |
| 6 | `scripts/freeze_audio_cohort.py` | Freezes the final cohort of tracks with usable audio |
| 7 | `scripts/run_phase3_refits.py` (uses `run_task3.py`, `run_task4.py`), `scripts/run_task6_clustering.py`, `scripts/run_phase3_cnn.py` | Trains the classical models, MLP, clustering and CNN on the frozen cohort |
| 8 | `scripts/run_phase4.py plan / validate / test / analyze` | Chooses thresholds on validation, locks them, scores the test set once, then computes metrics and bootstrap intervals |
| 9 | `scripts/report_phase4.py` | Builds the descriptive tables and figures in `results/phase4` |

## Reproducing the results

The analysis needs Python 3.13 or later.

**Check the final numbers (standard library only, a few seconds):**

```sh
python verify_results.py
```

This checks the file hashes, test membership and labels, then recomputes the confusion counts and hit F1 for all seven models under both threshold protocols.

**Recompute all test metrics and bootstrap intervals from the saved scores (no audio or GPU needed):**

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
python scripts/run_phase4.py analyze
python -m unittest discover -s tests
```

`analyze` rewrites `results/phase4/test_comparison.csv`, `TEST_UNCERTAINTY.json` and `PHASE4_SUMMARY.json` with identical values; only the completion timestamp changes.

**Full retraining from raw data:** download the metadata and audio as described above and run the pipeline in order. CNN training needs a CUDA GPU; the exact PyTorch build and hardware are recorded in `results/phase3/requirements-cnn-lock.txt` and `results/phase3/CNN_ENVIRONMENT.json`. Do not rerun `run_phase4.py test`: the test set was scored once, on 16 Nov 2026, and the saved predictions are the official result.

## Team

| Member | Main contribution |
|---|---|
| Taha Faisal | Model building and tuning (classical models, MLP and CNN) |
| Ibrahim Qasim | Data cleaning and duplicate rules |
| Ahmed Abdelwaly | Spectrogram conversion |
| Krisha Hingrajiya | Report writing and analysis |

## References

Defferrard, M., Benzi, K., Vandergheynst, P. and Bresson, X. (2017) "FMA: A Dataset for Music Analysis", *Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR)*. <https://github.com/mdeff/fma>
