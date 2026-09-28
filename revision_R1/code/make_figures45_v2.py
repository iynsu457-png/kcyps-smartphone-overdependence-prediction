import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
ROOT = os.path.dirname(REV)
BUNDLE = os.path.join(ROOT, "pipeline")
FIG = os.path.join(REV, "figures")
sys.path.insert(0, os.path.join(BUNDLE, "02_code"))
sys.path.insert(0, HERE)
import _paths
import revnumbers as NB
from grouped_shap_audit_labels import group_of

df = _paths.load_transitions()
feat = [c for c in df.columns if c not in {"ID", "WAVE", "TRANSITION", "y"}]
n_items = defaultdict(int)
for c in feat:
    g = group_of(c)
    if g is not None:
        n_items[g] += 1

def disp(g):
    k = n_items[g]
    return f"{NB.lab(g)} ({k})" if k > 1 else NB.lab(g)

AGG = pd.read_csv(os.path.join(BUNDLE, "03_results",
                               "shap_top25_xgboost_aggregated_filtered.csv"),
                  encoding="utf-8-sig", index_col=0).iloc[:, 0]
top = AGG.head(15)
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(10, 9.5))
yp = list(range(len(top)))[::-1]
plt.barh(yp, top.values, color="#3B75AF")
plt.yticks(yp, [disp(g) for g in top.index], fontsize=10)
plt.xlabel("Mean |SHAP value| (aggregated within scale)")
plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
plt.savefig(os.path.join(FIG, "Figure4_SHAP_bar_v2.png"), dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close()

ORDER5 = ["Parental mediation", "Psychological well-being", "Smartphone use",
          "Parenting stress", "Time use", "Family relationship", "Device access",
          "Aggression", "Parent school involvement", "Self-control",
          "School engagement", "Father age", "Social anxiety", "After-school activity",
          "Career orientation"]
src = Image.open(os.path.join(ROOT, "manuscript_PR", "figure 5_FINAL.png")).convert("RGBA")
bg = Image.new("RGBA", src.size, (255, 255, 255, 255))
bg.alpha_composite(src)
img = bg.convert("RGB")
a = np.asarray(img).astype(int)
dark = a.sum(axis=2) < 600
W, H = img.size
col = dark[:1030].sum(axis=0)
x_gap = next(x for x in range(300, 480) if col[x] == 0)
rows, inrun = [], False
for yy, v in enumerate(dark[:1040, :x_gap].sum(axis=1)):
    if v > 0 and not inrun:
        s, inrun = yy, True
    if v == 0 and inrun:
        rows.append((s, yy))
        inrun = False
assert len(rows) == len(ORDER5), rows
font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 27)
labels = [disp(g) for g in ORDER5]
meas = ImageDraw.Draw(img)
width = max(meas.textbbox((0, 0), t, font=font)[2] for t in labels)
L = width + 40
out = Image.new("RGB", (W - x_gap + L, H), "white")
out.paste(img.crop((x_gap, 0, W, H)), (L, 0))
dr = ImageDraw.Draw(out)
for (y0, y1), t in zip(rows, labels):
    dr.text((L - 12, (y0 + y1) / 2), t, font=font, fill=(51, 51, 51), anchor="rm")
out.save(os.path.join(FIG, "Figure5_SHAP_beeswarm_v2.png"), dpi=(300, 300))

print("Figure 4 rows:")
for g in top.index:
    print(f"  {disp(g):<50} {AGG[g]:.4f}")
print("Figure 5 rows:", "; ".join(labels))
print(f"written: {FIG}")
