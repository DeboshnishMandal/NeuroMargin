from pathlib import Path
import sys, re, json, warnings
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
 recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix,
 ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.neuromargin.audio import load_audio, extract_features

PAT=re.compile(r"^(ID\d+)_(pd|hc)_(\d+)_(\d+)_(\d+)\.wav$",re.I)

def main():
    # Create every output directory before feature extraction/model fitting.
    # This makes the script safe on fresh installs and after partial cleanup.
    (ROOT/"results"/"tables").mkdir(parents=True, exist_ok=True)
    (ROOT/"results"/"figures").mkdir(parents=True, exist_ok=True)
    (ROOT/"models").mkdir(parents=True, exist_ok=True)

    raw=ROOT/"data/raw"
    rows=[]; skipped=[]
    for p in sorted(raw.rglob("*.wav")):
        m=PAT.match(p.name)
        if not m: skipped.append(p.name); continue
        sid,label=m.group(1),m.group(2).lower()
        try:
            y,meta=load_audio(p)
            feat=extract_features(y)
            rows.append({"path":str(p),"participant":sid,"label":1 if label=="pd" else 0,**feat})
            print("Extracted",p.name)
        except Exception as e:
            print("Skipped",p.name,":",e)
    if len(rows)<20 or len({r["label"] for r in rows})<2:
        raise RuntimeError(f"Insufficient labeled audio. Extracted {len(rows)} recordings.")
    df=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).dropna()
    feature_cols=[c for c in df.columns if c not in {"path","participant","label"}]
    X=df[feature_cols].values; y=df.label.values; groups=df.participant.values
    n_groups=len(np.unique(groups))
    if n_groups<4: raise RuntimeError("Need at least four distinct participants for grouped validation.")
    n_splits=min(5,n_groups)
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=42)
    models={
      "RBF SVM":Pipeline([("scale",StandardScaler()),("model",SVC(C=1.0,kernel="rbf",class_weight="balanced",probability=True,random_state=42))]),
      "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced"))]),
      "Random Forest":RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1),
      "Extra Trees":ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1)
    }
    metrics=[]; oof={}
    for name,model in models.items():
        pred=cross_val_predict(model,X,y,groups=groups,cv=cv,method="predict",n_jobs=1)
        proba=cross_val_predict(model,X,y,groups=groups,cv=cv,method="predict_proba",n_jobs=1)[:,1]
        oof[name]=(pred,proba)
        metrics.append({"model":name,"accuracy":accuracy_score(y,pred),
          "balanced_accuracy":balanced_accuracy_score(y,pred),
          "precision":precision_score(y,pred,zero_division=0),
          "sensitivity":recall_score(y,pred,zero_division=0),
          "specificity":recall_score(y,pred,pos_label=0,zero_division=0),
          "f1":f1_score(y,pred,zero_division=0),
          "mcc":matthews_corrcoef(y,pred),
          "roc_auc":roc_auc_score(y,proba)})
        print(name,metrics[-1])
    mdf=pd.DataFrame(metrics).sort_values("balanced_accuracy",ascending=False)
    mdf.to_csv(ROOT/"results/tables/model_comparison.csv",index=False)
    # Primary deployment candidate: RBF SVM; fitted only after OOF evaluation.
    final=models["RBF SVM"].fit(X,y)
    joblib.dump({"model":final,"feature_columns":feature_cols,
      "training_participants":sorted(df.participant.unique().tolist()),
      "dataset":"MDVR-KCL (Zenodo 2867216)","label_map":{"0":"Healthy control","1":"Parkinson's disease"},
      "validation":"StratifiedGroupKFold participant-disjoint OOF; small cohort; exploratory"},ROOT/"models/voice_model.joblib")
    df.to_csv(ROOT/"results/tables/extracted_training_features.csv",index=False)
    pd.DataFrame([{"recordings":len(df),"participants":n_groups,"healthy_recordings":int((y==0).sum()),
      "pd_recordings":int((y==1).sum()),"features":len(feature_cols),"skipped_files":len(skipped)}]).to_csv(ROOT/"results/tables/dataset_summary.csv",index=False)
    # Plots
    figdir=ROOT/"results/figures"
    fig,ax=plt.subplots(figsize=(7,4)); df.label.value_counts().sort_index().rename({0:"Healthy",1:"PD"}).plot(kind="bar",ax=ax)
    ax.set_title("MDVR-KCL recording class counts"); ax.set_ylabel("Recordings"); ax.set_xlabel("Class"); fig.tight_layout(); fig.savefig(figdir/"class_distribution.png",dpi=180); plt.close(fig)
    best="RBF SVM"; pred,prob=oof[best]
    fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay(confusion_matrix(y,pred),display_labels=["Healthy","PD"]).plot(ax=ax,colorbar=False); ax.set_title("Participant-disjoint OOF confusion matrix"); fig.tight_layout(); fig.savefig(figdir/"confusion_matrix.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); RocCurveDisplay.from_predictions(y,prob,ax=ax); ax.plot([0,1],[0,1],"--",color="grey"); ax.set_title("RBF-SVM participant-disjoint OOF ROC"); fig.tight_layout(); fig.savefig(figdir/"roc_curve.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); mdf.set_index("model")[["balanced_accuracy","f1","roc_auc"]].plot(kind="bar",ax=ax); ax.set_ylim(0,1); ax.set_title("Model comparison (participant-disjoint OOF)"); fig.tight_layout(); fig.savefig(figdir/"model_comparison.png",dpi=180); plt.close(fig)
    # PCA is descriptive only, fitted on all features after evaluation.
    Xs=StandardScaler().fit_transform(X); pca=PCA(n_components=2,random_state=42); Z=pca.fit_transform(Xs)
    fig,ax=plt.subplots(figsize=(7,5))
    for lab,name in [(0,"Healthy"),(1,"PD")]:
        mask=y==lab; ax.scatter(Z[mask,0],Z[mask,1],label=name,alpha=.75)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"); ax.set_title("PCA of extracted acoustic features"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"pca_features.png",dpi=180); plt.close(fig)
    # RF feature importance descriptive.
    rf=models["Random Forest"].fit(X,y); imp=pd.Series(rf.feature_importances_,index=feature_cols).sort_values(ascending=False).head(15)
    fig,ax=plt.subplots(figsize=(8,6)); imp.sort_values().plot(kind="barh",ax=ax); ax.set_title("Top 15 Random Forest feature importances"); fig.tight_layout(); fig.savefig(figdir/"feature_importance.png",dpi=180); plt.close(fig)
    # CG convergence on regularized covariance system, independent numerical-method demo.
    A=(Xs.T@Xs)/len(Xs)+0.1*np.eye(Xs.shape[1]); b=np.ones(A.shape[0]); x=np.zeros_like(b); r=b-A@x; p=r.copy(); rs=float(r@r); residuals=[]
    for it in range(250):
        Ap=A@p; alpha=rs/(float(p@Ap)+1e-15); x=x+alpha*p; r=r-alpha*Ap; rs_new=float(r@r); residuals.append(np.sqrt(rs_new))
        if np.sqrt(rs_new)<1e-8: break
        p=r+(rs_new/(rs+1e-15))*p; rs=rs_new
    pd.DataFrame({"iteration":range(1,len(residuals)+1),"residual_norm":residuals}).to_csv(ROOT/"results/tables/cg_residuals.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,4)); ax.semilogy(range(1,len(residuals)+1),residuals); ax.set_title("Conjugate Gradient convergence"); ax.set_xlabel("Iteration"); ax.set_ylabel("Residual norm"); fig.tight_layout(); fig.savefig(figdir/"cg_convergence.png",dpi=180); plt.close(fig)
    print("Training complete. Model saved to models/voice_model.joblib")
if __name__=="__main__": main()
