"""
Camada de inferência: carrega o pipeline treinado UMA vez e valida/transforma as entradas.
O mesmo módulo é usado pela aplicação web (app.py) e pelos testes (tests/run_tests.py),
garantindo que a aplicação publicada e o código de inferência produzam resultados idênticos.
"""
import json
import os

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = joblib.load(os.path.join(BASE, "models", "pipeline.joblib"))
META = json.load(open(os.path.join(BASE, "models", "metadata.json"), encoding="utf-8"))
FEATURES = META["features"]

# Limites de validação (plausibilidade clínica; o treino cobre 0,08-80 anos, IMC 10-96 etc.)
LIMITS = {
    "age": (1, 100, "Idade (anos)"),
    "bmi": (10, 70, "IMC (kg/m²)"),
    "HbA1c_level": (3.5, 15, "HbA1c (%)"),
    "blood_glucose_level": (50, 500, "Glicose (mg/dL)"),
}
GENDERS = {"Female": "Feminino", "Male": "Masculino"}
SMOKING = {
    "never": "Nunca fumou",
    "former": "Ex-fumante",
    "current": "Fumante atual",
    "not current": "Não fuma atualmente",
    "ever": "Já fumou (sem detalhe)",
    "No Info": "Sem informação",
}


class ValidationError(ValueError):
    """Erro de validação com mensagens por campo."""

    def __init__(self, errors):
        super().__init__("; ".join(errors.values()))
        self.errors = errors


def _num(raw, key, errors):
    label = LIMITS[key][2]
    value = raw.get(key)
    if value is None or str(value).strip() == "":
        errors[key] = f"{label}: campo obrigatório."
        return None
    try:
        v = float(str(value).replace(",", "."))
    except ValueError:
        errors[key] = f"{label}: informe um número válido."
        return None
    if v != v or v in (float("inf"), float("-inf")):
        errors[key] = f"{label}: valor inválido."
        return None
    lo, hi, _ = LIMITS[key]
    if not lo <= v <= hi:
        errors[key] = f"{label}: valor fora do intervalo aceito ({lo} a {hi})."
        return None
    return v


def validate(raw: dict) -> dict:
    """Valida e converte a entrada bruta (strings do formulário). Lança ValidationError."""
    errors, clean = {}, {}
    for k in LIMITS:
        v = _num(raw, k, errors)
        if v is not None:
            clean[k] = v
    if raw.get("gender") not in GENDERS:
        errors["gender"] = "Sexo: selecione uma opção."
    else:
        clean["gender"] = raw["gender"]
    if raw.get("smoking_history") not in SMOKING:
        errors["smoking_history"] = "Histórico de tabagismo: selecione uma opção."
    else:
        clean["smoking_history"] = raw["smoking_history"]
    for k, label in [("hypertension", "Hipertensão"), ("heart_disease", "Doença cardíaca")]:
        if str(raw.get(k)) not in ("0", "1"):
            errors[k] = f"{label}: selecione Sim ou Não."
        else:
            clean[k] = int(raw[k])
    if errors:
        raise ValidationError(errors)
    return clean


def risk_band(p: float):
    """Converte probabilidade em faixa de linguagem simples."""
    if p < 0.10:
        return "baixo", "Baixa probabilidade de diabetes segundo o modelo."
    if p < 0.50:
        return "moderado", "Probabilidade intermediária: vale conversar com um profissional de saúde."
    return "alto", "Alta probabilidade de diabetes segundo o modelo: procure avaliação médica."


def predict(raw: dict) -> dict:
    """Uma inferência (sem re-treinar). Retorna classe, probabilidade, faixa e avisos."""
    d = validate(raw)
    X = pd.DataFrame([d], columns=FEATURES)
    proba = float(PIPELINE.predict_proba(X)[0, 1])
    label = int(proba >= 0.5)
    band, msg = risk_band(proba)
    warnings = []
    tr = META["ranges"]
    for k in LIMITS:
        lo, hi = tr[k]
        if not lo <= d[k] <= hi:
            warnings.append(f"{LIMITS[k][2]} = {d[k]:g} está fora da faixa vista no treinamento "
                            f"({lo:g}–{hi:g}); o resultado é menos confiável.")
    if d["age"] < 18:
        warnings.append("O modelo foi treinado majoritariamente com adultos; para crianças e adolescentes "
                        "o resultado é menos confiável.")
    return {"input": d, "probability": proba, "label": label, "band": band, "message": msg,
            "warnings": warnings}
