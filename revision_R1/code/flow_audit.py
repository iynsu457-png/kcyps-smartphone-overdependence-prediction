import io
import os
import sys
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", write_through=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
BUNDLE = os.path.join(os.path.dirname(REV), "pipeline")
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
import _paths

RAW = os.path.join(_paths.DATA_DIR, "raw")
PANELS = {"m1": ({7: 1, 8: 2, 9: 3, 10: 4}, [5, 6, 7]),
          "e4": ({7: 4, 8: 5, 9: 6, 10: 7}, [1, 2, 3])}
SPECIAL = {("e4", 3): "KCYPS2018e4Yw3_0907.sav", ("e4", 7): "KCYPS2018e4Yw7_0907.sav"}
REVERSE = [5, 10, 15]

def read_wave(panel, w):
    f = SPECIAL.get((panel, w), f"KCYPS2018{panel}Yw{w}.sav")
    path = os.path.join(RAW, f"KCYPS2018{panel}[SPSS]", f)
    _, meta = pyreadstat.read_sav(path, metadataonly=True)
    items = [f"YMDA1C{i:02d}w{w}" for i in range(1, 16)]
    part = f"SURVEY1w{w}"
    cols = ["ID"] + items + ([part] if part in meta.column_names else [])
    d, _ = pyreadstat.read_sav(path, usecols=cols)
    d.index = d["ID"].astype("int64")
    sc = d[items].copy()
    for i in REVERSE:
        sc[items[i - 1]] = 5 - sc[items[i - 1]]
    out = pd.DataFrame(index=d.index)
    out["sum"] = sc.sum(axis=1, skipna=False)
    out["daily"] = sc[items[:5]].sum(axis=1, skipna=False)
    out["participated"] = (d[part] == 1) if part in d else out["sum"].notna()
    return out

rows = []

def add(step, unit, panel, n, note=""):
    rows.append(dict(step=step, unit=unit, panel=panel, n=int(n), note=note))

waves, pairs = {}, []
for panel, (gmap, outside) in PANELS.items():
    for w in range(1, 8):
        waves[(panel, w)] = read_wave(panel, w)
    n_enr = len(waves[(panel, 1)])
    add("1 enrolled", "adolescents", panel, n_enr)
    slots = sum(len(waves[(panel, w)]) for w in range(1, 8))
    done = sum(int(waves[(panel, w)]["sum"].notna().sum()) for w in range(1, 8))
    add("2 wave records, all seven waves", "wave records (slots)", panel, slots)
    add("2 wave records, all seven waves", "wave records with completed SAPS", panel, done)
    o_slots = sum(len(waves[(panel, w)]) for w in outside)
    o_done = sum(int(waves[(panel, w)]["sum"].notna().sum()) for w in outside)
    add("3 excluded: outside the G7-G10 window", "wave records (slots)", panel, o_slots,
        f"waves {outside[0]}-{outside[-1]}")
    add("3 excluded: outside the G7-G10 window", "wave records with completed SAPS", panel,
        o_done)
    inw = [gmap[g] for g in (7, 8, 9, 10)]
    i_slots = sum(len(waves[(panel, w)]) for w in inw)
    i_done = sum(int(waves[(panel, w)]["sum"].notna().sum()) for w in inw)
    any_done = pd.concat([waves[(panel, w)]["sum"].notna() for w in inw], axis=1).any(axis=1)
    add("4 within-window records (G7-G10)", "wave records (slots)", panel, i_slots)
    add("4 within-window records (G7-G10)", "wave records with completed SAPS", panel, i_done)
    add("4 within-window records (G7-G10)", "adolescents with >= 1 completed SAPS", panel,
        int(any_done.sum()))
    for g in (7, 8, 9):
        cur, nxt = waves[(panel, gmap[g])], waves[(panel, gmap[g + 1])]
        idx = cur.index.union(nxt.index)
        P = pd.DataFrame(index=idx)
        P["panel"] = panel
        P["transition"] = f"G{g}->G{g + 1}"
        P["t_saps"] = cur["sum"].reindex(idx).notna()
        P["t1_sum"] = nxt["sum"].reindex(idx)
        P["t1_daily"] = nxt["daily"].reindex(idx)
        P["t1_part"] = nxt["participated"].reindex(idx).fillna(False).astype(bool)
        P["ID"] = idx
        pairs.append(P.reset_index(drop=True))

T = pd.concat(pairs, ignore_index=True)
miss = T["t1_sum"].isna()
high = ~miss & (T["t1_sum"] >= 45)
pot = ~miss & ~high & ((T["t1_sum"] >= 42) | (T["t1_daily"] >= 14))
T["stage"] = np.select([miss, pot], ["missing outcome", "potential risk"], "analytic")
T["y"] = high.astype(int)

for panel in ("m1", "e4", "all"):
    S = T if panel == "all" else T[T.panel == panel]
    m, p = S.stage == "missing outcome", S.stage == "potential risk"
    fs, an = S[~m], S[S.stage == "analytic"]
    add("5 candidate pairs (t, t+1)", "candidate pairs", panel, len(S))
    add("5 candidate pairs (t, t+1)", "adolescents", panel, S.ID.nunique())
    add("5 candidate pairs (t, t+1)", "pairs with completed SAPS at both waves", panel,
        int((S.t_saps & S.t1_sum.notna()).sum()))
    add("6 excluded: no outcome-wave SAPS", "candidate pairs", panel, int(m.sum()))
    add("6 excluded: no outcome-wave SAPS", "of which: no youth participation at t+1",
        panel, int((m & ~S.t1_part).sum()))
    add("6 excluded: no outcome-wave SAPS", "of which: participated, SAPS not completed",
        panel, int((m & S.t1_part).sum()))
    add("6 excluded: no outcome-wave SAPS", "adolescents with no remaining pair", panel,
        S.ID.nunique() - fs.ID.nunique())
    add("7 full-spectrum file", "transitions", panel, len(fs))
    add("7 full-spectrum file", "adolescents", panel, fs.ID.nunique())
    add("7 full-spectrum file", "transitions without predictor-wave SAPS", panel,
        int((~fs.t_saps).sum()))
    add("8 excluded: potential risk at t+1", "transitions", panel, int(p.sum()))
    add("8 excluded: potential risk at t+1", "adolescents with no remaining transition",
        panel, fs.ID.nunique() - an.ID.nunique())
    add("9 primary analytic file", "transitions", panel, len(an))
    add("9 primary analytic file", "adolescents", panel, an.ID.nunique())
    add("9 primary analytic file", "high-risk transitions", panel, int(an.y.sum()))
    add("9 primary analytic file", "transitions without predictor-wave SAPS", panel,
        int((~an.t_saps).sum()))
    per = an.groupby("ID").size().value_counts()
    for k in (1, 2, 3):
        add("9 primary analytic file", f"adolescents contributing {k} transition(s)", panel,
            int(per.get(k, 0)))
    for t in ("G7->G8", "G8->G9", "G9->G10"):
        add("9 primary analytic file", f"transitions {t}", panel,
            int((an.transition == t).sum()))

A = _paths.load_transitions()
an = T[T.stage == "analytic"]
key = lambda d: set(zip(d["ID"].astype("int64"), d["TRANSITION" if "TRANSITION" in d
                                                  else "transition"].astype(str)))
assert key(A) == key(an), "analytic keys differ from the frozen file"
assert (len(T), int(miss.sum()), int(pot.sum()), len(an), an.ID.nunique()) == \
    (15591, 1961, 2658, 10972, 4634)
ids = A["ID"].astype("int64").values
yv = A["y"].astype(int).values
dev, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
               .split(A, yv, groups=ids))
for part, idx in (("10 development set", dev), ("10 held-out test set", te)):
    add(part, "transitions", "all", len(idx))
    add(part, "adolescents", "all", len(np.unique(ids[idx])))
    add(part, "high-risk transitions", "all", int(yv[idx].sum()))

F = pd.DataFrame(rows)
F.to_csv(os.path.join(RES, "flow_audit.csv"), index=False, encoding="utf-8")
W = F.pivot_table(index=["step", "unit"], columns="panel", values="n", aggfunc="first",
                  sort=False)
print(W.to_string())
print(f"\nwritten: {os.path.join(RES, 'flow_audit.csv')}")
