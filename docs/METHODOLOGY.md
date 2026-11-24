# NeuroMargin methodology notes

## Why this build changes the audio model
The previous WAV demo extracted MFCCs from uploaded recordings but trained against UCI-489's pre-extracted acoustic columns. Matching column names do not imply matching feature definitions. This version trains and predicts using the same feature extractor on actual labeled WAV recordings.

## Data provenance and scope
MDVR-KCL contains 37 people (21 healthy controls and 16 PD) and recordings of read text and spontaneous dialogue. This is real-world audio and more suitable for a WAV pipeline than a feature-only CSV. Its small sample size makes estimates uncertain.

## Validation
All recordings from the same participant are assigned to one fold using StratifiedGroupKFold. Metrics are calculated on out-of-fold predictions. This reduces participant leakage but does not remove dataset confounding or guarantee external validity.

## Input scope
The upload page accepts WAV recordings of speech, but accepts technical format flexibility, not universal clinical generalization. Short isolated vowels, non-English speech, music, silence, noisy audio and recordings from different devices may be out of distribution. A future version should include an out-of-distribution detector and external cohort validation.

## Visualizations
- Confusion matrix, ROC curve, model comparison: out-of-fold evaluation.
- PCA: descriptive projection of extracted features.
- Random Forest importance: descriptive, not causal.
- Conjugate Gradient: separate numerical-method demonstration on a regularized covariance system; it is not the optimizer used by the deployed RBF SVM.
