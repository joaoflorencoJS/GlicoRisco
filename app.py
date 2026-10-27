"""
Aplicação web - Triagem de risco de diabetes.
Execução local:   python app.py            (http://127.0.0.1:5000)
Produção:         gunicorn app:app          (ver Dockerfile / README)

Cada requisição executa UMA inferência com o pipeline já treinado (carregado em predictor.py);
nenhum treinamento é refeito.

Observação sobre o uso de IA: a estrutura da aplicação e a interface foram geradas com apoio do
Claude (Anthropic) e revisadas pelo grupo. Veja o README (seção "Uso de IA").
"""
import os

from flask import Flask, jsonify, render_template, request

import predictor as P

app = Flask(__name__)

FIELD_HELP = [
    ("age", "Idade", "anos", "Idade da pessoa (1 a 100)."),
    ("gender", "Sexo", "", "Sexo biológico registrado."),
    ("bmi", "IMC", "kg/m²", "Índice de massa corporal = peso (kg) ÷ altura² (m²). Faixa aceita: 10 a 70."),
    ("HbA1c_level", "HbA1c", "%", "Hemoglobina glicada, média da glicose dos últimos ~3 meses. Faixa aceita: 3,5 a 15."),
    ("blood_glucose_level", "Glicose", "mg/dL", "Glicemia medida no exame. Faixa aceita: 50 a 500."),
    ("hypertension", "Hipertensão", "", "Diagnóstico prévio de pressão alta."),
    ("heart_disease", "Doença cardíaca", "", "Diagnóstico prévio de doença cardíaca."),
    ("smoking_history", "Tabagismo", "", "Histórico de tabagismo informado."),
]


def _ctx(**kw):
    base = dict(genders=P.GENDERS, smoking=P.SMOKING, limits=P.LIMITS, form={}, errors={}, result=None,
                meta=P.META, fields=FIELD_HELP)
    base.update(kw)
    return render_template("index.html", **base)


@app.get("/")
def index():
    return _ctx()


@app.post("/predict")
def predict_form():
    form = request.form.to_dict()
    try:
        result = P.predict(form)
        return _ctx(form=form, result=result)
    except P.ValidationError as e:
        return _ctx(form=form, errors=e.errors), 400


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True) or {}
    try:
        r = P.predict({k: payload.get(k) for k in P.FEATURES})
        return jsonify(r)
    except P.ValidationError as e:
        return jsonify({"errors": e.errors}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": P.META["modelo"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
