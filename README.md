# 🧠 NeuroMargin
### Voice-Based Parkinson's Disease Classification Using Acoustic Features and Machine Learning

**Course:** Mathematics for Computing III  
**Institution:** Amrita Vishwa Vidyapeetham, Bengaluru Campus  
**Team Members:** Soham and Karen

---

## 1. Project Overview

NeuroMargin is an experimental voice-based Parkinson's disease research prototype that investigates whether acoustic characteristics of speech can help distinguish recordings from individuals with Parkinson's disease (PD) and healthy controls.

The system extracts acoustic features from WAV recordings, evaluates multiple machine-learning classifiers, and presents experimental predictions through an interactive Streamlit dashboard.

**Note:** NeuroMargin is an academic research prototype, not a clinically validated diagnostic system.

## 2. Objectives

- Develop a consistent WAV audio preprocessing pipeline.
- Extract meaningful acoustic features from voice recordings.
- Compare multiple machine-learning classification algorithms.
- Evaluate models using participant-disjoint validation.
- Visualize acoustic characteristics and experimental results.
- Provide an interactive interface for WAV analysis.

## 3. Datasets

### MDVR-KCL Dataset
- Source: https://doi.org/10.5281/zenodo.2867216
- Contains voice recordings associated with Parkinson's disease and healthy controls.
- Used for WAV-based feature extraction and classification.

### UCI Parkinson Dataset with Replicated Acoustic Features
- Source: https://doi.org/10.24432/C5701F
- Contains 240 observations from 80 participants.
- Used for exploratory acoustic feature analysis and visualization.

The datasets are publicly available research resources and were not collected by the NeuroMargin team. Please consult their original licences and access conditions.

## 4. Technology Stack

- Python
- NumPy and Pandas
- Librosa and SoundFile
- Scikit-learn
- Matplotlib and Plotly
- Streamlit
- Joblib

## 5. Machine-Learning Models

The project evaluates:

1. RBF Support Vector Machine
2. Logistic Regression
3. Random Forest
4. Extra Trees

The conventional classifiers are implemented using Scikit-learn.

## 6. Audio Feature Extraction

The WAV processing pipeline includes:

- Stereo-to-mono conversion
- Resampling to 16 kHz
- Mel-frequency cepstral coefficients (MFCCs)
- Delta and delta-delta features
- RMS energy
- Zero-crossing rate
- Spectral centroid, bandwidth, rolloff and flatness
- Fundamental frequency statistics

## 7. Validation and Evaluation

The training workflow uses participant-disjoint stratified group cross-validation to reduce data leakage between recordings belonging to the same participant.

Evaluation metrics include:

- Accuracy
- Balanced accuracy
- Precision
- Sensitivity
- Specificity
- F1-score
- Matthews correlation coefficient
- ROC-AUC

## 8. Project Structure

```text
NeuroMargin/
├── app.py
├── requirements.txt
├── START_NEUROMARGIN.bat
├── FORCE_RETRAIN.bat
├── src/
│   └── neuromargin/
├── scripts/
├── data/
│   └── README.md
├── docs/
├── results/
└── tests/
```

## 9. Installation and Execution

1. Clone or download this repository.
2. Obtain the datasets using the instructions in `data/README.md`.
3. Ensure Python 3.13 is installed.
4. Run `START_NEUROMARGIN.bat`.
5. Allow the initial dependency installation and dataset preparation to complete.
6. Open the Streamlit dashboard in your browser when launched.

The launcher reuses the local environment, prepared dataset and trained model when their inputs have not changed.

## 10. Experimental Limitations

- The available datasets have limited participant diversity.
- Microphone quality and recording environments may affect acoustic features.
- Results may not generalize to other languages, accents or populations.
- Participant-disjoint validation does not replace independent external validation.
- Model predictions are experimental and must not be interpreted as medical diagnoses.

## 11. Team

**Soham**  
**Karen**

Both members contributed to the research and development of NeuroMargin. Refer to `docs/CONTRIBUTIONS.md` for the detailed contribution statement.

## 12. References

1. MDVR-KCL Dataset: https://doi.org/10.5281/zenodo.2867216
2. UCI Parkinson Dataset with Replicated Acoustic Features: https://doi.org/10.24432/C5701F
3. Cortes, C., and Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273–297.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
5. McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*.

---

**Repository:** NeuroMargin  
**Project Type:** Academic Research Prototype
