"""LightGBM re-ranker over the candidate table from features.build."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DROP = ["uid", "tv_show_id", "label", "cnt"]
SIZES = {"base": dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50),
         "big": dict(n_estimators=1000, learning_rate=0.03, num_leaves=127, min_child_samples=100)}


def params(size, seed):
    return dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=8,
                **SIZES[size])


def X(d):
    return d.drop(columns=[c for c in DROP if c in d])


def graded(d):
    """0 = not in truth; truth items 1..5 by log2 of their >=80 % airings in the target window."""
    return np.where(d.label > 0, 1 + np.minimum(4, np.floor(np.log2(np.maximum(d.cnt, 1)))), 0).astype(int)


def fit(d, obj="binary", seed=42, size="base"):
    """obj: binary (in truth top-5), poisson (count of >=80 % airings in the target window), lambdarank, lambdarank_graded."""
    if obj == "binary":
        m = lgb.LGBMClassifier(objective="binary", **params(size, seed))
        m.fit(X(d), d.label)
    elif obj == "poisson":
        m = lgb.LGBMRegressor(objective="poisson", **params(size, seed))
        m.fit(X(d), d.cnt)
    else:
        d = d.sort_values("uid", kind="stable")
        y = graded(d) if obj == "lambdarank_graded" else d.label
        m = lgb.LGBMRanker(objective="lambdarank", **params(size, seed))
        m.fit(X(d), y, group=d.groupby("uid", sort=False).size().values)
    return m


def predict(m, d):
    return m.predict_proba(X(d))[:, 1] if isinstance(m, lgb.LGBMClassifier) else m.predict(X(d))


def oof(d, obj="binary", seeds=(42,), extra=None, size="base"):
    """3-fold GroupKFold by user: every user is scored by a model that never saw them.
    extra: more training rows (fold B, uid + 10**6); rows of the held-out users are dropped."""
    p = np.zeros(len(d))
    for tr, te in GroupKFold(3).split(d, groups=d.uid):
        t = d.iloc[tr]
        if extra is not None:
            t = pd.concat([t, extra[~(extra.uid - 10**6).isin(set(d.uid.iloc[te]))]], ignore_index=True)
        for s in seeds:
            p[te] += predict(fit(t, obj, s, size), d.iloc[te]) / len(seeds)
    return p


def rank_avg(d, ps):
    """Ensemble: average of within-user percentile ranks of several score vectors."""
    return np.mean([pd.Series(p).groupby(d.uid.values).rank(pct=True).values for p in ps], axis=0)


def to_preds(d, p, k=5):
    t = pd.DataFrame({"uid": d.uid.values, "s": d.tv_show_id.values, "p": p})
    t = t.sort_values(["uid", "p"], ascending=[True, False]).groupby("uid").head(k)
    return t.groupby("uid").s.agg(list).to_dict()


def importance(m, d, n=10):
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X(d).columns)
    return (imp / imp.sum()).sort_values(ascending=False).head(n)
