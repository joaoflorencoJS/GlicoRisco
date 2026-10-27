"""
Análises complementares (rodar depois de train.py): estrutura dos rótulos, faixas de risco e limiares.
Gera results/extra.json e report/figures/{label_structure,risk_bands}.png.
Código escrito com apoio do Claude (Anthropic) e revisado pelo grupo.
"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

SEED = 42
meta = json.load(open("models/metadata.json", encoding="utf-8"))
FE = meta["features"]
df = pd.read_csv("data/diabetes_prediction_dataset.csv").drop_duplicates()
df = df[df.gender.isin(["Female", "Male"])].reset_index(drop=True)   # mesma limpeza do train.py
X, y = df[FE], df.diabetes
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)  # mesma divisão
model = joblib.load("models/pipeline.joblib")
p = model.predict_proba(Xte)[:, 1]
out = {}

# 1) Estrutura dos rótulos
hi = (df.HbA1c_level >= 6.8) | (df.blood_glucose_level >= 220)
lo = (df.HbA1c_level <= 5.0) | (df.blood_glucose_level <= 100)
mid = ~hi & ~lo
out["zonas"] = {
    "alta": {"n": int(hi.sum()), "taxa": float(df[hi].diabetes.mean())},
    "baixa": {"n": int(lo.sum()), "taxa": float(df[lo].diabetes.mean())},
    "intermediaria": {"n": int(mid.sum()), "taxa": float(df[mid].diabetes.mean()),
                      "positivos": int(df[mid].diabetes.sum())},
    "total_positivos": int(df.diabetes.sum()),
    "intermediaria_por_idade": {str(k): round(float(v), 3) for k, v in
                                df[mid].groupby(pd.cut(df[mid].age, [0, 30, 45, 60, 70, 81]), observed=True).diabetes.mean().items()},
}
rule = hi.astype(int)
out["regra_pura"] = {"precision": float(precision_score(df.diabetes, rule)),
                     "recall": float(recall_score(df.diabetes, rule)), "f1": float(f1_score(df.diabetes, rule))}

fig, ax = plt.subplots(1, 2, figsize=(8.4, 2.9))
g = df.groupby("HbA1c_level").diabetes.mean() * 100
ax[0].bar(range(len(g)), g.values, color=["#4C78A8" if v < 99 else "#E4572E" for v in g.values])
ax[0].set_xticks(range(len(g))); ax[0].set_xticklabels([f"{i:g}" for i in g.index], rotation=60, fontsize=7)
ax[0].set_title("% com diabetes por valor de HbA1c"); ax[0].set_xlabel("HbA1c (%)")
g = df.groupby("blood_glucose_level").diabetes.mean() * 100
ax[1].bar(range(len(g)), g.values, color=["#4C78A8" if v < 99 else "#E4572E" for v in g.values])
ax[1].set_xticks(range(len(g))); ax[1].set_xticklabels([f"{i:g}" for i in g.index], rotation=60, fontsize=7)
ax[1].set_title("% com diabetes por valor de glicose"); ax[1].set_xlabel("Glicose (mg/dL)")
plt.tight_layout(); plt.savefig("report/figures/label_structure.png", dpi=130); plt.close()

# 2) Faixas de risco (definidas a priori em predictor.py: <10%, 10-50%, >=50%) e taxa real observada no teste
bands = [("baixo (<10%)", 0, .10), ("moderado (10-50%)", .10, .50), ("alto (>=50%)", .50, 1.01)]
rows = []
for n, a, b in bands:
    m = (p >= a) & (p < b)
    rows.append({"faixa": n, "n": int(m.sum()), "pct_teste": float(m.mean()),
                 "taxa_real": float(yte[m].mean()), "prob_media_prevista": float(p[m].mean())})
out["faixas"] = rows
fig, ax = plt.subplots(figsize=(4.4, 2.9))
xs = np.arange(3)
ax.bar(xs - .2, [r["prob_media_prevista"] * 100 for r in rows], .38, color="#4C78A8", label="Prevista (média)")
ax.bar(xs + .2, [r["taxa_real"] * 100 for r in rows], .38, color="#E4572E", label="Observada")
ax.set_xticks(xs); ax.set_xticklabels(["Baixo", "Moderado", "Alto"]); ax.set_ylabel("% com diabetes")
ax.set_title("Faixas de risco: previsto x observado (teste)"); ax.legend(fontsize=7)
for i, r in enumerate(rows):
    ax.text(i, max(r["taxa_real"], r["prob_media_prevista"]) * 100 + 2, f"n={r['n']}", ha="center", fontsize=7)
ax.set_ylim(0, 115)
plt.tight_layout(); plt.savefig("report/figures/risk_bands.png", dpi=130); plt.close()

# 3) Efeito do limiar de decisão (apenas descritivo: o limiar de deploy continua 0,5)
th = []
for t in [.5, .3, .1]:
    pr = (p >= t).astype(int)
    th.append({"limiar": t, "precision": float(precision_score(yte, pr)), "recall": float(recall_score(yte, pr)),
               "f1": float(f1_score(yte, pr)), "sinalizados": int(pr.sum())})
out["limiares"] = th
json.dump(out, open("results/extra.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
