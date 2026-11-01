"""MAP@5 evaluator + the competition's ground-truth rule. Import or run as a self-test.

apk/mapk follow Kaggle's ml_metrics (github.com/benhamner/Metrics):
denominator min(|actual|, k); duplicates in the prediction are not re-counted.
"""
import numpy as np
import pandas as pd

K = 5


def apk(actual, predicted, k=K):
    actual = set(actual)
    if not actual:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for i, p in enumerate(list(predicted)[:k]):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1)
        seen.add(p)
    return score / min(len(actual), k)


def mapk(truth, preds, users, k=K):
    """truth, preds: dict user -> list. Users missing from truth score 0 (as on Kaggle)."""
    return float(np.mean([apk(truth.get(u, []), preds.get(u, []), k) for u in users]))


def make_truth(inter, thr=0.8, k=K, strict=False):
    """Competition rule (Data tab): per (user, airing) screen_time = overlap / duration;
    keep airings with screen_time >= thr; count them per tv_show_id; truth = k most frequent.
    inter columns: user_id, tv_show_id, ov (overlap seconds), dur (airing duration seconds).
    Tie-break by tv_show_id (host's rule unknown). strict=True keeps only users with exactly k shows.
    """
    t = inter[inter["ov"] >= thr * inter["dur"]]
    c = t.groupby(["user_id", "tv_show_id"]).size().rename("n").reset_index()
    c = c.sort_values(["user_id", "n", "tv_show_id"], ascending=[True, False, True])
    top = c.groupby("user_id").head(k)
    if strict:
        sizes = top.groupby("user_id").size()
        top = top[top["user_id"].isin(sizes[sizes == k].index)]
    return top.groupby("user_id")["tv_show_id"].agg(list).to_dict()


if __name__ == "__main__":
    # worked example from the lecture slide: relevant {A,B,C}
    assert abs(apk("ABC", "ABDCE") - 0.9167) < 1e-3
    assert abs(apk("ABC", "DAEBC") - 0.5333) < 1e-3
    assert apk("ABC", "AABBC") == apk("ABC", "AXBXC")  # duplicates don't double count
    assert apk([], "ABCDE") == 0.0
    # truth rule: user 1 watches show 10 fully 3 times, show 20 fully once, show 30 only 50 %
    df = pd.DataFrame({"user_id": [1] * 5, "tv_show_id": [10, 10, 10, 20, 30],
                       "ov": [60, 60, 60, 60, 30], "dur": [60, 60, 60, 60, 60]})
    assert make_truth(df)[1] == [10, 20]
    print("evaluate.py self-test OK")
