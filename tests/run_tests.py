"""
Casos de teste: mesma entrada na camada de inferência (predictor.py) e na aplicação web (HTTP real).
Uso: python tests/run_tests.py [URL_DA_APLICACAO]   (padrão: sobe a app local em http://127.0.0.1:5055)
Gera tests/results.json e capturas de tela em tests/screenshots/ (requer playwright, opcional).
"""
import json, os, re, subprocess, sys, time, urllib.parse, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import predictor as P

CASES = [
    {"id": "T1", "tipo": "válida", "nome": "Adulto jovem saudável",
     "esperado": "Risco baixo (exames normais, jovem).",
     "dados": dict(age="25", gender="Female", bmi="22,5", HbA1c_level="5.0", blood_glucose_level="90",
                   hypertension="0", heart_disease="0", smoking_history="never")},
    {"id": "T2", "tipo": "válida", "nome": "Exames claramente alterados",
     "esperado": "Risco alto (HbA1c 8,2% e glicose 240 mg/dL são valores típicos de diabetes).",
     "dados": dict(age="58", gender="Male", bmi="31.4", HbA1c_level="8.2", blood_glucose_level="240",
                   hypertension="1", heart_disease="0", smoking_history="former")},
    {"id": "T3", "tipo": "válida", "nome": "Idoso com fatores de risco e exames intermediários",
     "esperado": "Risco moderado (exames na zona cinzenta, idade e comorbidades elevam a probabilidade).",
     "dados": dict(age="68", gender="Male", bmi="34", HbA1c_level="6.2", blood_glucose_level="140",
                   hypertension="1", heart_disease="1", smoking_history="current")},
    {"id": "T4", "tipo": "inválida", "nome": "Campos vazios, texto e valor fora do intervalo",
     "esperado": "Nenhum cálculo; mensagens de erro claras por campo (HTTP 400).",
     "dados": dict(age="abc", gender="", bmi="500", HbA1c_level="", blood_glucose_level="140",
                   hypertension="0", heart_disease="", smoking_history="never")},
    {"id": "T5", "tipo": "difícil", "nome": "Falso negativo real do conjunto de teste (linha 16327)",
     "esperado": "Rótulo real conhecido: TEM diabetes. Exames na zona intermediária; o modelo tende a subestimar o risco.",
     "dados": dict(age="41", gender="Female", bmi="27.8", HbA1c_level="6.1", blood_glucose_level="155",
                   hypertension="0", heart_disease="0", smoking_history="never")},
    {"id": "T6", "tipo": "difícil", "nome": "Falso positivo real do conjunto de teste (linha 25570)",
     "esperado": "Rótulo real conhecido: NÃO tem diabetes. IMC muito alto + doença cardíaca + 62 anos elevam o risco estimado.",
     "dados": dict(age="62", gender="Female", bmi="44.81", HbA1c_level="6.2", blood_glucose_level="130",
                   hypertension="0", heart_disease="1", smoking_history="No Info")},
    {"id": "T7", "tipo": "limítrofe", "nome": "Criança (população pouco representada no treino)",
     "esperado": "Sem verdade conhecida. Esperado: resultado acompanhado de aviso de menor confiabilidade (idade < 18).",
     "dados": dict(age="9", gender="Male", bmi="17", HbA1c_level="5.0", blood_glucose_level="90",
                   hypertension="0", heart_disease="0", smoking_history="No Info")},
]


def http_post(url, data):
    body = urllib.parse.urlencode(data).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, body), timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main():
    base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else None
    proc = None
    if base is None:
        base = "http://127.0.0.1:5055"
        env = dict(os.environ, PORT="5055")
        proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "app.py")], env=env, cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(40):
            try:
                urllib.request.urlopen(base + "/health", timeout=1); break
            except Exception:
                time.sleep(.5)
    out = []
    try:
        for c in CASES:
            rec = dict(id=c["id"], tipo=c["tipo"], nome=c["nome"], entrada=c["dados"], esperado=c["esperado"])
            try:
                r = P.predict(c["dados"])
                rec["inferencia_local"] = {"probabilidade": round(r["probability"], 6), "faixa": r["band"],
                                           "classe": r["label"], "avisos": r["warnings"]}
            except P.ValidationError as e:
                rec["inferencia_local"] = {"erros": e.errors}
            status, html = http_post(base + "/predict", c["dados"])
            rec["http_status"] = status
            m = re.search(r'class="score">([\d.]+)%', html)
            rec["app_probabilidade_pct"] = float(m.group(1)) if m else None
            rec["app_erros"] = re.findall(r'class="msg">([^<]+)<', html)
            loc = rec["inferencia_local"]
            rec["consistente"] = (("erros" in loc and status == 400 and len(rec["app_erros"]) == len(loc["erros"]))
                                  or ("probabilidade" in loc and status == 200 and
                                      abs(round(loc["probabilidade"] * 100, 1) - rec["app_probabilidade_pct"]) < 0.051))
            out.append(rec)
            print(c["id"], c["tipo"], status, rec.get("app_probabilidade_pct"), "consistente:", rec["consistente"])
        json.dump(out, open(os.path.join(ROOT, "tests", "results.json"), "w"), indent=2, ensure_ascii=False)

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("playwright ausente: capturas de tela puladas"); return
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 1100, "height": 900})
            pg.goto(base + "/")
            pg.screenshot(path=os.path.join(ROOT, "tests", "screenshots", "00_inicial.png"), full_page=True)
            for c in CASES:
                pg.goto(base + "/")
                for k, v in c["dados"].items():
                    pg.fill(f"#{k}", v) if k in ("age", "bmi", "HbA1c_level", "blood_glucose_level") else pg.select_option(f"#{k}", v) if v else None
                pg.click("button.primary")
                pg.wait_for_load_state()
                pg.screenshot(path=os.path.join(ROOT, "tests", "screenshots", f"{c['id']}.png"), full_page=True)
            b.close()
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
