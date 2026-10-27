"""
Treinamento e avaliação - Previsão de diabetes (dataset Kaggle "Diabetes prediction dataset").

Uso:  python train.py
Saídas: models/pipeline.joblib, models/metadata.json, report/figures/*.png, results/metrics.json

Observação sobre o uso de IA: a estrutura inicial deste script foi gerada com apoio do Claude
(Anthropic) e revisada pelo grupo. Veja o README (seção "Uso de IA").
"""
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve, accuracy_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
SEED = 42
DATA_PATH = "data/diabetes_prediction_dataset.csv"
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("report/figures", exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10})
C_NEG, C_POS = "#4C78A8", "#E4572E"

# ----------------------------------------------------------------------------------------------
# 1. Carga e análise inicial
# ----------------------------------------------------------------------------------------------
raw = pd.read_csv(DATA_PATH)
info = {"registros_brutos": int(len(raw)), "atributos": int(raw.shape[1] - 1),
        "ausentes": int(raw.isna().sum().sum()), "duplicatas": int(raw.duplicated().sum()),
        "gender_other": int((raw.gender == "Other").sum()),
        "prevalencia_bruta": float(raw.diabetes.mean())}
print("Análise inicial:", info)

# ----------------------------------------------------------------------------------------------
# 2. Limpeza (feita ANTES da divisão: remover duplicatas evita que a mesma linha caia em treino e teste)
# ----------------------------------------------------------------------------------------------
df = raw.drop_duplicates().copy()
df = df[df.gender.isin(["Female", "Male"])].copy()  # 'Other' tem só 18 linhas: insuficiente para aprender
df = df.reset_index(drop=True)
info["registros_limpos"] = int(len(df))
info["prevalencia_limpa"] = float(df.diabetes.mean())
print(f"Após limpeza: {len(df)} registros; prevalência {df.diabetes.mean():.3%}")

NUM = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BIN = ["hypertension", "heart_disease"]
CAT = ["gender", "smoking_history"]
FEATURES = NUM + BIN + CAT
X, y = df[FEATURES], df["diabetes"]

# ----------------------------------------------------------------------------------------------
# 3. Divisão: teste (20%) separado e usado apenas no final; escolha de configuração por CV no treino
# ----------------------------------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
print(f"Treino: {len(X_tr)} | Teste: {len(X_te)}")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def prep(scale: bool):
    """Pré-processamento ajustado apenas no treino (dentro do Pipeline/CV)."""
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUM),
        ("bin", "passthrough", BIN),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), CAT),
    ])


candidates = {
    "Regressão Logística": (
        Pipeline([("prep", prep(True)), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        {"clf__C": [0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}),
    "Árvore de Decisão": (
        Pipeline([("prep", prep(False)), ("clf", DecisionTreeClassifier(random_state=SEED))]),
        {"clf__max_depth": [2, 3, 4, 5, 6, 8, 12, None], "clf__min_samples_leaf": [1, 20],
         "clf__class_weight": [None, "balanced"]}),
    "Random Forest": (
        Pipeline([("prep", prep(False)),
                  ("clf", RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=SEED))]),
        {"clf__max_depth": [6, 10], "clf__min_samples_leaf": [5],
         "clf__class_weight": [None, "balanced"]}),
}

# ----------------------------------------------------------------------------------------------
# 4. Treinamento e comparação (critério de seleção: F1 da classe "diabetes" na validação cruzada)
# ----------------------------------------------------------------------------------------------
results, best_models, grids = {}, {}, {}
for name, (pipe, grid) in candidates.items():
    gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    grids[name] = gs
    best_models[name] = gs.best_estimator_
    print(f"{name}: melhor F1 (CV) = {gs.best_score_:.4f} | {gs.best_params_}")
    results[name] = {"cv_f1": float(gs.best_score_),
                     "params": {k: (None if v is None else v) for k, v in gs.best_params_.items()}}

# Variação de profundidade da árvore (para o gráfico de comparação de configurações)
depth_rows = []
for d in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
    m = Pipeline([("prep", prep(False)), ("clf", DecisionTreeClassifier(max_depth=d, random_state=SEED))])
    f1_cv = cross_val_predict(m, X_tr, y_tr, cv=cv, n_jobs=-1)
    m.fit(X_tr, y_tr)
    depth_rows.append({"depth": d, "f1_cv": f1_score(y_tr, f1_cv), "f1_train": f1_score(y_tr, m.predict(X_tr))})
depth_df = pd.DataFrame(depth_rows)

# ----------------------------------------------------------------------------------------------
# 5. Avaliação final no teste (tocado uma única vez por modelo, após a escolha das configurações)
# ----------------------------------------------------------------------------------------------
test_rows = {}
for name, model in best_models.items():
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_rows[name] = {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "precision_neg": precision_score(y_te, pred, pos_label=0),
        "recall_neg": recall_score(y_te, pred, pos_label=0),
        "f1_neg": f1_score(y_te, pred, pos_label=0),
        "cm": confusion_matrix(y_te, pred).tolist(),
    }
    results[name]["test"] = {k: (float(v) if not isinstance(v, list) else v) for k, v in test_rows[name].items()}
    print(f"\n=== {name} (teste) ===")
    print(classification_report(y_te, pred, target_names=["sem diabetes", "diabetes"], digits=3))

# ----------------------------------------------------------------------------------------------
# 6. Escolha do modelo para deploy
#    Regra definida a priori: maior F1 (CV) da classe positiva; em empate (<0.005) preferir o mais simples/interpretável.
# ----------------------------------------------------------------------------------------------
complexity = {"Regressão Logística": 0, "Árvore de Decisão": 1, "Random Forest": 2}
ranked = sorted(results, key=lambda n: -results[n]["cv_f1"])
top = ranked[0]
chosen = top
for n in ranked[1:]:
    if results[top]["cv_f1"] - results[n]["cv_f1"] < 0.005 and complexity[n] < complexity[chosen]:
        chosen = n
print("\nModelo escolhido para deploy:", chosen)
results["_escolhido"] = chosen

# ----------------------------------------------------------------------------------------------
# 7. Figuras para o relatório
# ----------------------------------------------------------------------------------------------
# 7.1 Distribuição da classe e variáveis-chave
fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
cnt = df.diabetes.value_counts().sort_index()
ax[0].bar(["Sem diabetes", "Diabetes"], cnt.values, color=[C_NEG, C_POS])
for i, v in enumerate(cnt.values):
    ax[0].text(i, v, f"{v:,}\n({v/len(df):.1%})".replace(",", "."), ha="center", va="bottom", fontsize=8)
ax[0].set_ylim(0, cnt.max() * 1.25); ax[0].set_title("Distribuição da variável-alvo")
for k, c, lab in [(0, C_NEG, "Sem diabetes"), (1, C_POS, "Diabetes")]:
    ax[1].hist(df[df.diabetes == k].HbA1c_level, bins=25, alpha=.7, color=c, label=lab, density=True)
    ax[2].hist(df[df.diabetes == k].blood_glucose_level, bins=25, alpha=.7, color=c, label=lab, density=True)
ax[1].set_title("HbA1c (%) por classe"); ax[1].legend(fontsize=7)
ax[2].set_title("Glicose em jejum (mg/dL) por classe")
plt.tight_layout(); plt.savefig("report/figures/eda.png"); plt.close()

# 7.2 Taxa de diabetes por faixa de idade / IMC
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
age_bins = pd.cut(df.age, [0, 20, 30, 40, 50, 60, 70, 81])
r = df.groupby(age_bins, observed=True).diabetes.mean() * 100
ax[0].bar(range(len(r)), r.values, color=C_NEG); ax[0].set_xticks(range(len(r)))
ax[0].set_xticklabels(["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"], fontsize=8)
ax[0].set_title("% com diabetes por faixa etária")
bmi_bins = pd.cut(df.bmi, [0, 18.5, 25, 30, 35, 100])
r = df.groupby(bmi_bins, observed=True).diabetes.mean() * 100
ax[1].bar(range(len(r)), r.values, color=C_NEG); ax[1].set_xticks(range(len(r)))
ax[1].set_xticklabels(["<18,5", "18,5-25", "25-30", "30-35", ">35"], fontsize=8)
ax[1].set_title("% com diabetes por faixa de IMC")
plt.tight_layout(); plt.savefig("report/figures/eda_rates.png"); plt.close()

# 7.3 Comparação de profundidade da árvore
fig, ax = plt.subplots(figsize=(4.6, 3))
ax.plot(depth_df.depth, depth_df.f1_train, "o-", color=C_POS, label="Treino")
ax.plot(depth_df.depth, depth_df.f1_cv, "o-", color=C_NEG, label="Validação cruzada (5 folds)")
ax.set_xlabel("max_depth"); ax.set_ylabel("F1 (classe diabetes)"); ax.legend(fontsize=8)
ax.set_title("Árvore de Decisão: efeito da profundidade")
plt.tight_layout(); plt.savefig("report/figures/tree_depth.png"); plt.close()

# 7.4 Matrizes de confusão (teste)
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for a, (name, m) in zip(ax, best_models.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_te, (m.predict_proba(X_te)[:, 1] >= .5).astype(int)),
                           display_labels=["Sem diab.", "Diabetes"]).plot(ax=a, cmap="Blues", colorbar=False,
                                                                        values_format="d")
    a.set_title(name); a.grid(False)
plt.tight_layout(); plt.savefig("report/figures/confusion_matrices.png"); plt.close()

# 7.5 ROC e Precisão-Recall
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
for (name, m), c in zip(best_models.items(), ["#4C78A8", "#E4572E", "#54A24B"]):
    p = m.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p); ax[0].plot(fpr, tpr, color=c, label=f"{name} ({roc_auc_score(y_te, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, p); ax[1].plot(rc, pr, color=c, label=f"{name} ({average_precision_score(y_te, p):.3f})")
ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=.8); ax[0].set_xlabel("Taxa de falsos positivos"); ax[0].set_ylabel("Recall")
ax[0].set_title("Curva ROC (AUC)"); ax[0].legend(fontsize=7, loc="lower right")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precisão"); ax[1].set_title("Curva Precisão-Recall (AP)")
ax[1].legend(fontsize=7, loc="lower left")
plt.tight_layout(); plt.savefig("report/figures/roc_pr.png"); plt.close()

# 7.6 Importância das variáveis do modelo escolhido (permutação no teste)
from sklearn.inspection import permutation_importance
best = best_models[chosen]
pi = permutation_importance(best, X_te, y_te, scoring="f1", n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
names_pt = {"HbA1c_level": "HbA1c", "blood_glucose_level": "Glicose", "age": "Idade", "bmi": "IMC",
            "hypertension": "Hipertensão", "heart_disease": "Doença cardíaca", "gender": "Sexo",
            "smoking_history": "Hist. tabagismo"}
fig, ax = plt.subplots(figsize=(4.6, 3))
ax.barh([names_pt[i] for i in imp.index], imp.values, color=C_NEG)
ax.set_xlabel("Queda do F1 ao embaralhar a variável"); ax.set_title(f"Importância por permutação ({chosen})")
plt.tight_layout(); plt.savefig("report/figures/importance.png"); plt.close()
results["importancia_permutacao"] = {k: float(v) for k, v in imp.items()}

# 7.7 Calibração do modelo escolhido
from sklearn.calibration import calibration_curve
p = best.predict_proba(X_te)[:, 1]
fr, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(3.6, 3))
ax.plot([0, 1], [0, 1], "--", color="gray", lw=.8); ax.plot(mp, fr, "o-", color=C_NEG)
ax.set_xlabel("Probabilidade prevista"); ax.set_ylabel("Fração real com diabetes"); ax.set_title("Calibração (teste)")
plt.tight_layout(); plt.savefig("report/figures/calibration.png"); plt.close()

# 7.8 Estrutura da árvore (se escolhida) - útil para explicar o modelo
if chosen == "Árvore de Decisão":
    clf = best.named_steps["clf"]
    fn = list(best.named_steps["prep"].get_feature_names_out())
    fig, ax = plt.subplots(figsize=(10, 4.2))
    plot_tree(clf, feature_names=fn, class_names=["Sem diab.", "Diabetes"], filled=True, rounded=True,
              max_depth=3, fontsize=6, ax=ax, impurity=False, proportion=True)
    plt.tight_layout(); plt.savefig("report/figures/tree.png"); plt.close()

# ----------------------------------------------------------------------------------------------
# 8. Erros do modelo escolhido: onde ele falha?
# ----------------------------------------------------------------------------------------------
pred = (p >= .5).astype(int)
err = X_te.copy(); err["y"] = y_te.values; err["pred"] = pred
fn_ = err[(err.y == 1) & (err.pred == 0)]; fp_ = err[(err.y == 0) & (err.pred == 1)]
results["analise_erros"] = {
    "falsos_negativos": int(len(fn_)), "falsos_positivos": int(len(fp_)),
    "fn_hba1c_medio": float(fn_.HbA1c_level.mean()) if len(fn_) else None,
    "fn_glicose_media": float(fn_.blood_glucose_level.mean()) if len(fn_) else None,
    "fp_hba1c_medio": float(fp_.HbA1c_level.mean()) if len(fp_) else None,
    "fp_glicose_media": float(fp_.blood_glucose_level.mean()) if len(fp_) else None,
    "fn_idade_media": float(fn_.age.mean()) if len(fn_) else None,
    "fp_idade_media": float(fp_.age.mean()) if len(fp_) else None,
}
print("\nAnálise de erros:", results["analise_erros"])

# ----------------------------------------------------------------------------------------------
# 9. Re-treino do modelo escolhido com TODOS os dados limpos? NÃO: mantém-se o modelo treinado só no treino,
#    para que as métricas do relatório correspondam exatamente ao artefato publicado.
# ----------------------------------------------------------------------------------------------
joblib.dump(best, "models/pipeline.joblib")
meta = {
    "modelo": chosen, "features": FEATURES, "sklearn_version": sklearn.__version__,
    "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUM},
    "categorias": {"gender": ["Female", "Male"], "smoking_history": sorted(df.smoking_history.unique().tolist())},
    "prevalencia_treino": float(y_tr.mean()),
    "metricas_teste": results[chosen]["test"],
}
json.dump(meta, open("models/metadata.json", "w"), indent=2, ensure_ascii=False)
json.dump({"info": info, "results": results,
           "depth": depth_df.round(4).to_dict(orient="records"),
           "split": {"treino": int(len(X_tr)), "teste": int(len(X_te)),
                     "pos_treino": int(y_tr.sum()), "pos_teste": int(y_te.sum())}},
          open("results/metrics.json", "w"), indent=2, ensure_ascii=False, default=str)
print("\nArtefatos salvos.")
