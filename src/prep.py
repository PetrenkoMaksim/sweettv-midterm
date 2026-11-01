"""Join views with the labelled schedule -> cache/*.parquet, results/data_stats.txt, results/join_checks.txt."""
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KIT_ROOT", Path(__file__).resolve().parents[1]))
DATA, CACHE, RES = ROOT / "data", ROOT / "cache", ROOT / "results"
BIG = 10**10  # packs (channel, unix seconds) into one sortable int64 key
SCHED_FMT = "%d.%m.%Y %H:%M:%S"


def secs(t):
    return (t - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")


def iso_week(sec):
    return pd.to_datetime(sec, unit="s").dt.isocalendar().week.astype("int16").values


def load_views():
    f = sorted(glob.glob(str(DATA / "dataset11-30*.csv")))[0]
    v = pd.read_csv(f, dtype={"user_id": str})
    raw = v.copy()
    v["start"] = secs(pd.to_datetime(v.start_time, format="ISO8601"))
    v["stop"] = secs(pd.to_datetime(v.stop_time, format="ISO8601"))
    return raw, v


def load_schedule():
    parts = []
    for f in sorted(glob.glob(str(DATA / "export_arh_*.csv"))):
        s = pd.read_csv(f, low_memory=False)
        s["file"] = Path(f).name
        parts.append(s)
    s = pd.concat(parts, ignore_index=True)
    s["start"] = secs(pd.to_datetime(s.start_time, format=SCHED_FMT))
    s["dur"] = s.duration.astype("int64")
    s["end"] = s.start + s.dur
    return s


def pairs(v, s):
    """All (view row, airing row) pairs on the same channel with positive overlap. s sorted by channel, start."""
    ks = s.channel_id.values * BIG + s.start.values
    ke = np.maximum.accumulate(s.channel_id.values * BIG + s.end.values)
    c = v.vsetv_id.values.astype("int64") * BIG
    lo = np.searchsorted(ke, c + v.start.values, "right")  # first airing that may still be running
    hi = np.searchsorted(ks, c + v.stop.values, "left")  # airings starting before the view stops
    n = np.clip(hi - lo, 0, None)
    vi = np.repeat(np.arange(len(v)), n)
    ai = np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    ov = np.minimum(v.stop.values[vi], s.end.values[ai]) - np.maximum(v.start.values[vi], s.start.values[ai])
    m = ov > 0
    return vi[m], ai[m], ov[m]


def tz_check(v, s, out):
    """Share of view starts within 60 s of an airing start on the same channel, for shifts -3..+3 h."""
    ks = np.sort(s.channel_id.values * BIG + s.start.values)
    smp = v.sample(min(500_000, len(v)), random_state=42)
    c = smp.vsetv_id.values.astype("int64") * BIG
    out.append("shift_h  share_|d|<=60s  share_|d|<=300s  median_|d|_s")
    for h in range(-3, 4):
        q = c + smp.start.values + h * 3600
        i = np.clip(np.searchsorted(ks, q), 1, len(ks) - 1)
        d = np.minimum(np.abs(q - ks[i - 1]), np.abs(ks[i] - q))
        out.append(f"{h:+d}  {np.mean(d <= 60):.4f}  {np.mean(d <= 300):.4f}  {np.median(d):.0f}")


def main():
    t0 = time.time()
    CACHE.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    raw, v = load_views()
    s = load_schedule()
    tpl = pd.read_csv(DATA / "submission.csv", dtype=str)
    st = []
    st.append(f"views file columns: {list(raw.columns)}\n{raw.dtypes.to_string()}")
    st.append(f"views rows {len(v):,}  users {v.user_id.nunique():,}  channels {v.vsetv_id.nunique():,}")
    st.append(f"views start_time examples: {raw.start_time.head(3).tolist()}")
    st.append(f"views start range {pd.to_datetime(v.start.min(), unit='s')} .. {pd.to_datetime(v.stop.max(), unit='s')}")
    vw = iso_week(v.start)
    st.append(f"views ISO weeks {vw.min()}..{vw.max()}; rows per week:\n{pd.Series(vw).value_counts().sort_index().to_string()}")
    st.append(f"duraton == stop-start: {np.mean(v.duraton.values == v.stop.values - v.start.values):.4f}; "
              f"duraton quantiles {v.duraton.quantile([0, .5, .9, .99, 1]).tolist()}")
    st.append(f"views with stop<=start: {(v.stop <= v.start).sum()}")
    st.append(f"schedule columns: {list(s.columns)}")
    sw = iso_week(s.start)
    s["week"] = sw
    for f, g in s.groupby("file"):
        lab = g.tv_show_id != 0
        st.append(f"{f}: rows {len(g):,}, labelled rows {lab.sum():,} ({lab.mean():.3f}), labelled airtime share "
                  f"{g.dur[lab].sum() / g.dur.sum():.3f}, shows {g.tv_show_id[lab].nunique():,}, "
                  f"channels {g.channel_id.nunique():,}, ISO weeks {g.week.min()}..{g.week.max()}")
    st.append(f"template rows {len(tpl):,}, columns {list(tpl.columns)}, example {tpl.iloc[0].tolist()}")
    st.append(f"template users present in views: {tpl.user_id.isin(set(v.user_id)).sum():,}")
    st.append(f"view channels present in schedule: {np.isin(v.vsetv_id.unique(), s.channel_id.unique()).mean():.3f}")

    # users -> int codes (view users + template users)
    users = pd.DataFrame({"user_id": sorted(set(v.user_id) | set(tpl.user_id))})
    users["uid"] = np.arange(len(users), dtype="int32")
    users["test"] = users.user_id.isin(set(tpl.user_id))
    users.to_parquet(CACHE / "users.parquet")
    v["uid"] = v.user_id.map(dict(zip(users.user_id, users.uid))).astype("int32")

    # cleaned schedule (all rows for checks, labelled rows for items)
    s = s.sort_values(["channel_id", "start"]).reset_index(drop=True)
    jc = []
    vi, ai, ov = pairs(v, s)
    tot = (v.stop - v.start).clip(lower=0).sum()
    lab = s.tv_show_id.values[ai] != 0
    jc.append(f"watched seconds total {tot:,}; landing on any airing {ov.sum() / tot:.4f}; "
              f"on labelled airtime {ov[lab].sum() / tot:.4f}; views touching a labelled airing "
              f"{len(np.unique(vi[lab])) / len(v):.4f}")
    tz_check(v, s, jc)

    lab_s = s[s.tv_show_id != 0].reset_index(drop=True)
    lab_s["aid"] = np.arange(len(lab_s), dtype="int32")
    keep = ["aid", "channel_id", "tv_show_id", "start", "end", "dur", "week", "tv_show_title", "tv_show_category",
            "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production"]
    keep = [c for c in keep if c in lab_s.columns]
    lab_s[keep].to_parquet(CACHE / "schedule.parquet")

    vi, ai, ov = pairs(v, lab_s)
    p = pd.DataFrame({"uid": v.uid.values[vi], "aid": ai.astype("int32"), "ov": ov})
    it = p.groupby(["uid", "aid"], as_index=False).ov.sum()
    a = lab_s.loc[it.aid]
    it["tv_show_id"] = a.tv_show_id.values
    it["channel_id"] = a.channel_id.values
    it["dur"] = a.dur.values
    it["start"] = a.start.values
    it["week"] = a.week.values
    it["ov_raw"] = it.ov
    it["ov"] = np.minimum(it.ov, it.dur)
    it["frac"] = it.ov / it.dur.clip(lower=1)
    it.to_parquet(CACHE / "interactions.parquet")
    jc.append(f"interactions (user, labelled airing): {len(it):,}; clipped by duration: {(it.ov_raw > it.dur).sum():,}; "
              f"users {it.uid.nunique():,}; shows {it.tv_show_id.nunique():,}")
    jc.append(f"overlap seconds quantiles (5,25,50,75,95,99%): {it.ov.quantile([.05, .25, .5, .75, .95, .99]).round(0).tolist()}")
    jc.append(f"frac quantiles (5,25,50,75,95%): {it.frac.quantile([.05, .25, .5, .75, .95]).round(3).tolist()}; "
              f"share frac>=0.8: {(it.frac >= .8).mean():.4f}")
    jc.append(f"airing durations with dur<=0: {(lab_s.dur <= 0).sum()}")

    vv = pd.DataFrame({"uid": v.uid.values, "channel_id": v.vsetv_id.values, "start": v.start.values,
                       "stop": v.stop.values, "week": vw})
    vv.to_parquet(CACHE / "views.parquet")

    # one worked example for the hand check: a view that matches >= 2 labelled airings
    cnt = pd.Series(vi).value_counts()
    ex = int(cnt[cnt >= 2].index.sort_values()[len(cnt[cnt >= 2]) // 2])
    m = vi == ex
    r = raw.iloc[ex]
    hc = ["## (ii) One view record and the labelled airings it matched (from src/prep.py)", "",
          f"View record (row {ex} of the views file): user_id={r.user_id}, vsetv_id={r.vsetv_id}, "
          f"start_time={r.start_time}, stop_time={r.stop_time}, duraton={r.duraton}", "",
          "| tv_show_id | title | airing start | airing end | duration s | overlap = min(ends) − max(starts) |",
          "|---|---|---|---|---|---|"]
    for j, o in zip(ai[m], ov[m]):
        x = lab_s.iloc[j]
        hc.append(f"| {x.tv_show_id} | {x.tv_show_title} | {pd.to_datetime(x.start, unit='s')} | "
                  f"{pd.to_datetime(x.end, unit='s')} | {x.dur} | {o} s (frac of airing {o / x.dur:.3f}) |")
    (RES / "hand_check_join.md").write_text("\n".join(hc) + "\n")
    (RES / "data_stats.txt").write_text("\n".join(st) + "\n")
    (RES / "join_checks.txt").write_text("\n".join(jc) + "\n")
    print("\n".join(st + jc + hc))
    print(f"prep done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
