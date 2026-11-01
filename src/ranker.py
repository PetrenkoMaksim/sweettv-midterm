"""LightGBM re-ranker over the candidate table from features.build."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DROP = ["uid", "tv_show_id", "label"]


def params(obj, seed):
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=8)
    return p


def fit(d, obj="binary", seed=42):
    X = d.drop(columns=[c for c in DROP if c in d])
    if obj == "binary":
        m = lgb.LGBMClassifier(objective="binary", **params(obj, seed))
        m.fit(X, d.label)
    else:
        d = d.sort_values("uid", kind="stable")
        m = lgb.LGBMRanker(objective="lambdarank", **params(obj, seed))
        m.fit(d.drop(columns=[c for c in DROP if c in d]), d.label, group=d.groupby("uid", sort=False).size().values)
    return m


def predict(m, d):
    X = d.drop(columns=[c for c in DROP if c in d])
    return m.predict_proba(X)[:, 1] if isinstance(m, lgb.LGBMClassifier) else m.predict(X)


def oof(d, obj="binary", seeds=(42,)):
    """3-fold GroupKFold by user: every user is scored by a model that never saw them."""
    p = np.zeros(len(d))
    for tr, te in GroupKFold(3).split(d, groups=d.uid):
        for s in seeds:
            p[te] += predict(fit(d.iloc[tr], obj, s), d.iloc[te]) / len(seeds)
    return p


def to_preds(d, p, k=5):
    t = pd.DataFrame({"uid": d.uid.values, "s": d.tv_show_id.values, "p": p})
    t = t.sort_values(["uid", "p"], ascending=[True, False]).groupby("uid").head(k)
    return t.groupby("uid").s.agg(list).to_dict()


def importance(m, d, n=10):
    X = d.drop(columns=[c for c in DROP if c in d])
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    return (imp / imp.sum()).sort_values(ascending=False).head(n)
