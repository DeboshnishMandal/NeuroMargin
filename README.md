# NeuroMargin — Full Voice Research Project

A reproducible WAV-based Parkinson's voice research prototype. The project downloads a real, labeled WAV corpus on first run, extracts one consistent acoustic feature vector for both training and inference, trains and evaluates multiple models using participant-disjoint folds, creates graphs, and launches a Streamlit dashboard.

## One-command start (Windows)

1. Install **64-bit Python 3.13** and ensure `py --list` shows it.
2. Extract this ZIP.
3. Double-click `START_NEUROMARGIN.bat`.

The first run installs packages, downloads the MDVR-KCL archive (~606 MB), verifies its MD5, extracts the recordings, computes features, trains models, generates graphs and opens the dashboard. It requires internet access and disk space for the archive plus extracted WAVs and features. Later starts reuse the local dataset and trained model.

## Dataset

**MDVR-KCL — Mobile Device Voice Recordings at King's College London**  
Zenodo record: https://zenodo.org/records/2867216  
Direct archive: https://zenodo.org/records/2867216/files/26_29_09_2017_KCL.zip?download=1  
DOI: https://doi.org/10.5281/zenodo.2867216  
License: CC BY 4.0 (cite the source).

This corpus includes 37 participants (21 healthy controls, 16 PD) with WAV speech recordings. The project parses labels and participant IDs from filenames and performs grouped validation. This is a small research corpus, not a representative clinical population.

## Project output

- `models/voice_model.joblib` — trained SVM and feature schema
- `results/tables/model_comparison.csv` — OOF model metrics
- `results/tables/extracted_training_features.csv` — computed training features
- `results/figures/` — class distribution, confusion matrix, ROC, model comparison, PCA, feature importance, CG convergence
- Streamlit dashboard — upload a WAV and view an experimental binary model output plus extracted features.

## Method

- Audio input: WAV, mono or stereo, at least 2 seconds, up to 120 seconds.
- Resampling: 16 kHz mono.
- Features: MFCCs and deltas, spectral centroid/bandwidth/rolloff/flatness, RMS, zero-crossing rate, pitch statistics, silence fraction and duration.
- Models: RBF-SVM (deployment candidate), Logistic Regression, Random Forest, Extra Trees.
- Validation: StratifiedGroupKFold, grouped by participant. Recordings from a participant cannot be split across training and test folds.

## Important limitations

This cannot reliably diagnose Parkinson's disease. The MDVR-KCL dataset has only 37 participants and its recordings are English speech from a particular acquisition context. A model trained on this cohort may not generalize to arbitrary languages, microphones, ages, accents, environments, sustained vowels, short clips or unrelated audio. The displayed class and probability are experimental model outputs, not clinical probabilities. Do not use them for medical decisions.

A universal classifier for any audio sample cannot be guaranteed from a small corpus. The model requires speech that is reasonably similar to its training data. For scientifically stronger results, validate externally on independent, demographically diverse cohorts and use clinician-reviewed labels.

## Retraining

Delete `models/voice_model.joblib` and run `START_NEUROMARGIN.bat` again. Or activate `.venv` and run `python scripts/train.py`.

## Citation

Jaeger, H., Trivedi, D., & Stadtschnitzer, M. (2019). *Mobile Device Voice Recordings at King's College London (MDVR-KCL) from both early and advanced Parkinson's disease patients and healthy controls*. Zenodo. https://doi.org/10.5281/zenodo.2867216

Also retain the original UCI-489 dataset in `data/raw/` if you want to compare the earlier acoustic-feature experiments; it is not used to train the WAV classifier.


## Additional UCI-489 feature analysis
The bundled UCI-489 CSV is analyzed separately to produce feature variance, feature correlation, PCA, parameter boxplots and participant-grouped model metrics under `results/uci489/`. These tabular-feature results are **not** used to train the WAV classifier; the WAV classifier is trained only on MDVR-KCL raw WAVs so training and inference share the same feature extractor.

## Fast relaunch behavior (incremental setup)

Use `START_NEUROMARGIN.bat` for both first launch and later launches.

- **First launch:** creates `.venv`, installs requirements, downloads/verifies and extracts MDVR-KCL if it is not already present, generates the UCI-489 analysis, trains the voice model, then opens Streamlit. The dataset download and first training can take substantial time.
- **Later launches:** reuses the virtual environment, cached dataset, analysis outputs, and trained model. It skips pip installation, dataset preparation, UCI analysis, audio feature extraction, and model training when their tracked inputs have not changed; it goes straight to Streamlit.
- If `requirements.txt`, the analysis script/input CSV, or model training/audio pipeline code changes, only the affected stage is rerun.
- Run `FORCE_RETRAIN.bat` when you deliberately want to retrain the voice model with the already-prepared dataset.
- `.setup` contains cache fingerprints. If you manually replace or edit the extracted WAV dataset, delete `.setup\mdvr_kcl_ready.json` and `.setup\voice_model.json` before launching so the dataset manifest and model are rebuilt.

The fast path assumes the extracted research dataset is unchanged after preparation. It deliberately avoids rescanning hundreds of WAV files on every launch.
