from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
csv=ROOT/"data/raw/ReplicatedAcousticFeatures-ParkinsonDatabase.csv"
if not csv.exists():
    print("UCI-489 CSV not found; skipping its feature-only analysis.")
    raise SystemExit(0)
df=pd.read_csv(csv)
# UCI-489 replicated acoustic feature database: preserve participant grouping.
id_col=next((c for c in df.columns if c.lower() in {"id","subject","subject_id"}),None)
status_col=next((c for c in df.columns if c.lower()=="status"),None)
if status_col is None:
    raise RuntimeError("Could not locate Status label in UCI-489 CSV.")
groups=df[id_col].astype(str).values if id_col else np.arange(len(df)).astype(str)
y=df[status_col].astype(int).values
drop=[status_col]+([id_col] if id_col else [])
Xdf=df.drop(columns=drop).select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
Xdf=Xdf.fillna(Xdf.median(numeric_only=True))
X=Xdf.values
out=ROOT/"results/uci489"; figs=out/"figures"; tabs=out/"tables"
figs.mkdir(parents=True,exist_ok=True); tabs.mkdir(parents=True,exist_ok=True)
pd.DataFrame({"class":["Healthy","PD"],"recordings":[int((y==0).sum()),int((y==1).sum())]}).to_csv(tabs/"class_counts.csv",index=False)
# Participant-disjoint grouped evaluation
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
models={"RBF SVM":make_pipeline(StandardScaler(),SVC(C=1,kernel="rbf",class_weight="balanced",probability=True,random_state=42)),
        "Random Forest":RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42)}
metrics=[]
for name,model in models.items():
    pred=cross_val_predict(model,X,y,groups=groups,cv=cv,method="predict")
    prob=cross_val_predict(model,X,y,groups=groups,cv=cv,method="predict_proba")[:,1]
    metrics.append({"model":name,"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),
                    "f1":f1_score(y,pred),"roc_auc":roc_auc_score(y,prob)})
pd.DataFrame(metrics).to_csv(tabs/"model_metrics.csv",index=False)
# Top variance acoustic feature distributions and correlation
variances=Xdf.var().sort_values(ascending=False)
top=variances.head(12).index.tolist()
fig,ax=plt.subplots(figsize=(9,5)); variances.head(15).sort_values().plot(kind="barh",ax=ax)
ax.set_title("UCI-489: highest-variance acoustic parameters"); fig.tight_layout(); fig.savefig(figs/"feature_variance.png",dpi=180); plt.close(fig)
corr=Xdf[top].corr()
fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1); ax.set_xticks(range(len(top)),top,rotation=90); ax.set_yticks(range(len(top)),top); fig.colorbar(im,ax=ax); ax.set_title("UCI-489: acoustic feature correlation"); fig.tight_layout(); fig.savefig(figs/"feature_correlation.png",dpi=180); plt.close(fig)
# PCA descriptive
Xs=StandardScaler().fit_transform(X); Z=PCA(n_components=2,random_state=42).fit_transform(Xs)
fig,ax=plt.subplots(figsize=(7,5))
for k,label in [(0,"Healthy"),(1,"PD")]:
    m=y==k; ax.scatter(Z[m,0],Z[m,1],label=label,alpha=.7)
ax.set_title("UCI-489: PCA of replicated acoustic features"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); fig.tight_layout(); fig.savefig(figs/"pca.png",dpi=180); plt.close(fig)
# Per-parameter distribution: top 6 variance features
fig,axes=plt.subplots(2,3,figsize=(13,7))
for ax,col in zip(axes.ravel(),top[:6]):
    ax.boxplot([Xdf.loc[y==0,col],Xdf.loc[y==1,col]],tick_labels=["Healthy","PD"],showfliers=False)
    ax.set_title(col); ax.tick_params(axis="x",labelrotation=15)
fig.suptitle("UCI-489: selected acoustic parameter distributions"); fig.tight_layout(); fig.savefig(figs/"parameter_boxplots.png",dpi=180); plt.close(fig)
print("UCI-489 feature-only analysis saved under results/uci489.")
