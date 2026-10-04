import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
RES = os.environ.get("REV_RESULTS", os.path.join(REV, "results"))
OUT = os.path.join(REV, "figures", "Figure1_Flow_v2.png")

F = pd.read_csv(os.path.join(RES, "flow_audit.csv"))

def n(step, unit, panel="all"):
    s = F[F.step.str.startswith(step) & (F.unit == unit)]
    if panel == "all" and "all" not in set(s.panel):
        return int(s.n.sum())
    return int(s[s.panel == panel].n.iloc[0])

def f(x):
    return f"{x:,}"

enr = {p: n("1 ", "adolescents", p) for p in ("m1", "e4")}
slots = {p: n("2 ", "wave records (slots)", p) for p in ("m1", "e4")}
done = {p: n("2 ", "wave records with completed SAPS", p) for p in ("m1", "e4")}
out_s = {p: n("3 ", "wave records (slots)", p) for p in ("m1", "e4")}
out_d = n("3 ", "wave records with completed SAPS")
in_s, in_d = n("4 ", "wave records (slots)"), n("4 ", "wave records with completed SAPS")
in_a = n("4 ", "adolescents with >= 1 completed SAPS")
cp, cp_a = n("5 ", "candidate pairs"), n("5 ", "adolescents")
mo = n("6 ", "candidate pairs")
mo_np = n("6 ", "of which: no youth participation at t+1")
mo_ns = n("6 ", "of which: participated, SAPS not completed")
mo_a = n("6 ", "adolescents with no remaining pair")
fs, fs_a = n("7 ", "transitions"), n("7 ", "adolescents")
pr, pr_a = n("8 ", "transitions"), n("8 ", "adolescents with no remaining transition")
an, an_a = n("9 ", "transitions"), n("9 ", "adolescents")
hr = n("9 ", "high-risk transitions")
nopred = n("9 ", "transitions without predictor-wave SAPS")
k1, k2, k3 = (n("9 ", f"adolescents contributing {k} transition(s)") for k in (1, 2, 3))
tr = [n("9 ", f"transitions {t}") for t in ("G7->G8", "G8->G9", "G9->G10")]
dev = {u: n("10 development", u) for u in ("transitions", "adolescents",
                                             "high-risk transitions")}
tst = {u: n("10 held-out", u) for u in ("transitions", "adolescents",
                                          "high-risk transitions")}
hi_fs = hr
gen_fs = fs - pr - hi_fs
assert cp == 3 * cp_a and sum(slots.values()) == 7 * cp_a
assert fs == cp - mo and an == fs - pr

STY = {"src": ("#EEEEEE", "#B5B5B5"), "main": ("#D6E0EE", "#8FA8C8"),
       "excl": ("#EFD9DC", "#C99BA2"), "part": ("#CFE8DC", "#7FBF9F")}
FONT = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(11, 15.2))
ax.set_xlim(0, 110)
ax.set_ylim(0, 152)
ax.axis("off")

def box(xc, yc, w, h, lines, kind, bold_first=True, fs=9.6):
    fc, ec = STY[kind]
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                boxstyle="round,pad=0.2,rounding_size=0.8",
                                fc=fc, ec=ec, lw=1.5, zorder=2))
    step = 2.35
    y0 = yc + (len(lines) - 1) * step / 2
    for i, t in enumerate(lines):
        ax.text(xc, y0 - i * step, t, ha="center", va="center", fontsize=fs,
                family=FONT, zorder=3,
                fontweight="bold" if (i == 0 and bold_first) else "normal")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.8, color="#5A5A7A", zorder=1))

XM, XE, WM, WE = 38, 91, 58, 36

box(18.5, 145, 36, 11.5, [
    "KCYPS 2018 m1 panel",
    "(middle-school Grade-1 cohort, from 2018)",
    f"Enrolled: {f(enr['m1'])} adolescents",
    f"Wave records, waves 1–7: {f(slots['m1'])}",
    f"  of which completed SAPS: {f(done['m1'])}"], "src", fs=9.0)
box(57.5, 145, 36, 11.5, [
    "KCYPS 2018 e4 panel",
    "(elementary Grade-4 cohort, from 2018)",
    f"Enrolled: {f(enr['e4'])} adolescents",
    f"Wave records, waves 1–7: {f(slots['e4'])}",
    f"  of which completed SAPS: {f(done['e4'])}"], "src", fs=9.0)
ax.plot([18.5, 18.5, 57.5, 57.5], [139.0, 136.8, 136.8, 139.0], color="#5A5A7A", lw=1.8,
        zorder=1)
arrow(XM, 136.8, XM, 133.4)

box(XM, 127.5, WM, 11, [
    "Pooled panels, grade-harmonized to G7–G10",
    f"Adolescents: {f(cp_a)}",
    f"Wave records: {f(sum(slots.values()))} (7 waves × {f(cp_a)})",
    f"  of which completed SAPS: {f(sum(done.values()))}"], "main")
box(XE, 127.5, WE, 11, [
    "Excluded: wave records outside G7–G10",
    f"{f(sum(out_s.values()))} records",
    f"(m1 waves 5–7: {f(out_s['m1'])}; e4 waves 1–3: {f(out_s['e4'])})",
    f"  of which completed SAPS: {f(out_d)}",
    "No adolescent excluded"], "excl", fs=8.6)
arrow(XM + WM / 2 + 0.3, 127.5, XE - WE / 2 - 0.4, 127.5)
arrow(XM, 121.8, XM, 117.4)

box(XM, 111.8, WM, 10.5, [
    "Within-window wave records (G7, G8, G9, G10)",
    f"Wave records: {f(in_s)} (4 grades × {f(cp_a)})",
    f"  of which completed SAPS: {f(in_d)}",
    f"Adolescents with ≥ 1 completed SAPS: {f(in_a)}"], "main")
arrow(XM, 106.4, XM, 101.9)

box(XM, 96.2, WM, 10.5, [
    "Candidate pairs (grade t → t+1)",
    f"Pairs: {f(cp)} (3 per adolescent × {f(cp_a)})",
    f"G7→G8, G8→G9, G9→G10: {f(cp_a)} each",
    f"Adolescents: {f(cp_a)}"], "main")
box(XE, 96.2, WE, 12.5, [
    "Excluded: no SAPS at the outcome wave",
    f"{f(mo)} pairs",
    f"  no youth participation at t+1: {f(mo_np)}",
    f"  participated, SAPS not completed: {f(mo_ns)}",
    f"Adolescents left with no pair: {f(mo_a)}"], "excl", fs=8.6)
arrow(XM + WM / 2 + 0.3, 96.2, XE - WE / 2 - 0.4, 96.2)
arrow(XM, 90.8, XM, 86.3)

box(XM, 80.6, WM, 10.5, [
    "Full-spectrum file (sensitivity analyses)",
    f"Transitions: {f(fs)}   Adolescents: {f(fs_a)}",
    f"Outcome at t+1: general use {f(gen_fs)},",
    f"potential risk {f(pr)}, high risk {f(hi_fs)}"], "main")
box(XE, 80.6, WE, 10.5, [
    "Excluded: potential risk at t+1",
    f"{f(pr)} transitions",
    "(retained in the sensitivity analyses)",
    f"Adolescents left with no transition: {f(pr_a)}"], "excl", fs=8.6)
arrow(XM + WM / 2 + 0.3, 80.6, XE - WE / 2 - 0.4, 80.6)
arrow(XM, 75.2, XM, 70.2)

box(XM, 62.6, WM, 14.5, [
    "Primary analytic file",
    f"Transitions: {f(an)}   Adolescents: {f(an_a)}",
    f"High risk at t+1: {f(hr)} ({100 * hr / an:.2f} %)",
    f"G7→G8: {f(tr[0])} | G8→G9: {f(tr[1])} | G9→G10: {f(tr[2])}",
    f"Adolescents with 1 / 2 / 3 transitions: {f(k1)} / {f(k2)} / {f(k3)}",
    f"Transitions without a predictor-wave SAPS: {f(nopred)}"], "main")

arrow(XM, 55.2, XM, 50.0)
ax.plot([20, 58], [50.0, 50.0], color="#5A5A7A", lw=1.8, zorder=1)
arrow(20, 50.0, 20, 45.6)
arrow(58, 50.0, 58, 45.6)
ax.text(XM + 1.2, 52.6, "Split by adolescent (GroupShuffleSplit, seed 42, 20 % held out)",
        ha="left", va="center", fontsize=8.8, style="italic", family=FONT,
        color="#374151")
for xc, title, d, gpos in ((20, "Development set", dev, "dev"),
                           (58, "Held-out test set", tst, "test")):
    box(xc, 38.4, 34, 13, [
        title,
        f"Transitions: {f(d['transitions'])}",
        f"Adolescents: {f(d['adolescents'])}",
        f"High risk at t+1: {f(d['high-risk transitions'])} "
        f"({100 * d['high-risk transitions'] / d['transitions']:.2f} %)",
        f"General use at t+1: {f(d['transitions'] - d['high-risk transitions'])}"],
        "part")

ax.set_ylim(29, 152)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"written: {OUT}")
