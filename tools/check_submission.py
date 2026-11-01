"""Validate a Kaggle submission file before uploading (uploads are limited!).

usage: python tools/check_submission.py submissions/xxx.csv data/submission.csv data/export_arh_31-42-final.csv
"""
import sys
import pandas as pd

sub_path, template_path, sched_path = sys.argv[1:4]
sub = pd.read_csv(sub_path, dtype=str)
tpl = pd.read_csv(template_path, dtype=str)
sched = pd.read_csv(sched_path, usecols=["tv_show_id"], dtype={"tv_show_id": "int64"})
valid = set(sched.loc[sched.tv_show_id != 0, "tv_show_id"].astype(str))

ok = True
def check(cond, msg):
    global ok
    if not cond:
        ok = False
        print("FAIL:", msg)

check(list(sub.columns) == ["user_id", "tv_show_id"], f"columns are {list(sub.columns)}")
check(len(sub) == len(tpl), f"{len(sub)} rows, template has {len(tpl)}")
check(sub.user_id.tolist() == tpl.user_id.tolist(), "user_id order/values differ from template")
lists = sub.tv_show_id.fillna("").str.split()
check((lists.str.len() == 5).all(), "every row must have exactly 5 ids")
check(lists.map(lambda l: len(set(l)) == len(l)).all(), "duplicate ids inside a row")
check(lists.map(lambda l: "0" not in l).all(), "id 0 (unlabelled) present")
unknown = lists.map(lambda l: [x for x in l if x not in valid])
check(unknown.map(len).sum() == 0, f"{unknown.map(len).sum()} ids not in the weeks 31-42 schedule")
print("OK" if ok else "NOT OK", "-", sub_path)
sys.exit(0 if ok else 1)
