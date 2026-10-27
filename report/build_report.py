# -*- coding: utf-8 -*-
"""Gera relatorio.pdf e casos_de_teste.pdf (na raiz) a partir de results/*.json, figuras e capturas.
Script escrito com apoio do Claude (Anthropic) e revisado pelo grupo."""
import json, os
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle, NextPageTemplate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
FD = "/usr/share/fonts/truetype/liberation/"
pdfmetrics.registerFont(TTFont("L", FD + "LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("L-B", FD + "LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("L-I", FD + "LiberationSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("L-BI", FD + "LiberationSans-BoldItalic.ttf"))
pdfmetrics.registerFont(TTFont("M", FD + "LiberationMono-Regular.ttf"))
pdfmetrics.registerFontFamily("L", normal="L", bold="L-B", italic="L-I", boldItalic="L-BI")

M = json.load(open(os.path.join(ROOT, "results/metrics.json"), encoding="utf-8"))
X = json.load(open(os.path.join(ROOT, "results/extra.json"), encoding="utf-8"))
T = json.load(open(os.path.join(ROOT, "tests/results.json"), encoding="utf-8"))
R = M["results"]

TEAL, INK, GREY = colors.HexColor("#0f6b7a"), colors.HexColor("#16222b"), colors.HexColor("#5b6b74")
LINE = colors.HexColor("#cfd9de")
body = ParagraphStyle("body", fontName="L", fontSize=9.4, leading=12.6, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=4)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.2, alignment=TA_LEFT, spaceAfter=0)
cap = ParagraphStyle("cap", parent=body, fontSize=8, leading=10, textColor=GREY, alignment=TA_LEFT, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="L-B", fontSize=12.5, leading=15, textColor=TEAL, spaceBefore=9, spaceAfter=4, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="L-B", fontSize=10, leading=12.5, textColor=INK, spaceBefore=5, spaceAfter=2, keepWithNext=1)
bul = ParagraphStyle("bul", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)
cell = ParagraphStyle("cell", fontName="L", fontSize=7.8, leading=9.6, textColor=INK, alignment=TA_LEFT)
cellb = ParagraphStyle("cellb", parent=cell, fontName="L-B")
ttl = ParagraphStyle("ttl", fontName="L-B", fontSize=25, leading=30, textColor=INK, alignment=TA_LEFT)
sub = ParagraphStyle("sub", fontName="L", fontSize=12.5, leading=17, textColor=GREY, alignment=TA_LEFT)
cov = ParagraphStyle("cov", fontName="L", fontSize=10.5, leading=15, textColor=INK, alignment=TA_LEFT)


def P(t, s=body): return Paragraph(t, s)
def B(t): return Paragraph(t, bul, bulletText="•")
def fmtn(x): return f"{x:,}".replace(",", ".")
def pct(x, d=1): return f"{x*100:.{d}f}%".replace(".", ",")
def f3(x): return f"{x:.3f}".replace(".", ",")
def fig(name, w_cm):
    from PIL import Image as PI
    w, h = PI.open(os.path.join(FIG, name)).size
    return Image(os.path.join(FIG, name), width=w_cm * cm, height=w_cm * cm * h / w)
def tbl(data, widths, header=True, zebra=True, fs=None):
    rows = [[c if not isinstance(c, str) else Paragraph(c, cellb if (header and i == 0) else cell) for c in r]
            for i, r in enumerate(data)]
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), .4, LINE),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header: st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eef1")), ("LINEBELOW", (0, 0), (-1, 0), .8, TEAL)]
    t.setStyle(TableStyle(st)); return t


def footer(c, d):
    c.saveState(); c.setFont("L", 7.5); c.setFillColor(GREY)
    c.drawString(2 * cm, 1.2 * cm, "GlicoRisco – Tópicos Avançados em Inteligência Artificial")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {d.page}"); c.restoreState()


def make(path, story, first_page_footer=True):
    d = BaseDocTemplate(path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="GlicoRisco – Relatório", author="Grupo GlicoRisco")
    fr = Frame(d.leftMargin, d.bottomMargin, d.width, d.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    d.addPageTemplates([PageTemplate(id="cover", frames=[fr]), PageTemplate(id="main", frames=[fr], onPage=footer)])
    d.build(story)


# ===================================================================================== RELATÓRIO
tr, te, info = R["Árvore de Decisão"], None, M["info"]
S = []
# ---- Capa
S += [Spacer(1, 4.2 * cm), P("GlicoRisco", ttl), Spacer(1, .2 * cm),
      P("Triagem de risco de diabetes com aprendizado de máquina: preparação dos dados, comparação de modelos e deploy de uma aplicação web", sub),
      Spacer(1, 1.2 * cm),
      P("<b>Disciplina:</b> Tópicos Avançados em Inteligência Artificial<br/><b>Professor:</b> Prof. Dr. Márcio Andrey Teixeira<br/>"
        "<b>Entrega:</b> 26/10/2026", cov), Spacer(1, .6 * cm),
      P("<b>Integrantes:</b><br/>[NOME COMPLETO DO INTEGRANTE 1] – [matrícula]<br/>[NOME COMPLETO DO INTEGRANTE 2] – [matrícula]<br/>"
        "[NOME COMPLETO DO INTEGRANTE 3] – [matrícula]", cov), Spacer(1, .6 * cm),
      P("<b>Aplicação publicada:</b> [COLE AQUI O LINK DA APLICAÇÃO APÓS O DEPLOY]<br/><b>Repositório (opcional):</b> [LINK DO REPOSITÓRIO]", cov),
      Spacer(1, 1.2 * cm),
      P("Este trabalho utilizou ferramentas de IA generativa (Claude, da Anthropic) como apoio. Os pontos exatos de uso estão sinalizados ao longo do "
        "texto com a marca <b>[IA]</b> e detalhados na seção 8.", small),
      NextPageTemplate("main"), PageBreak()]

# ---- 1 e 2
S += [P("1. Identificação", h1),
      P("<b>Projeto:</b> GlicoRisco – triagem de risco de diabetes. <b>Integrantes:</b> [NOME 1], [NOME 2], [NOME 3] (ver contribuições na seção 8). "
        "<b>Tipo de problema:</b> classificação binária supervisionada (tem / não tem diabetes), com saída em probabilidade."),
      P("2. Problema e objetivo", h1),
      P("O diabetes mellitus costuma ser diagnosticado tarde porque a pessoa só procura exames quando já há sintomas. Uma triagem que use informações "
        "simples e exames de rotina pode indicar quem deve priorizar a confirmação laboratorial. O objetivo do projeto é treinar, avaliar e publicar um "
        "modelo que recebe os dados de uma pessoa e devolve a <b>probabilidade estimada de diabetes</b>, uma <b>faixa de risco</b> (baixo, moderado ou alto) "
        "e uma classificação binária."),
      P("<b>Público-alvo:</b> estudantes e profissionais de saúde em contexto educacional e pessoas que têm em mãos resultados de exames e querem "
        "entender o que eles indicam. <b>Entradas:</b> idade, sexo, IMC, HbA1c, glicose, hipertensão, doença cardíaca e histórico de tabagismo. "
        "<b>Resultado:</b> percentual de risco, faixa e mensagem em linguagem simples. <b>Como ajuda:</b> indica se vale buscar avaliação médica. A ferramenta "
        "<b>não é diagnóstica</b> e isso é informado na própcia interface.".replace("própcia", "própria"))]

# ---- 3 Dataset
attrs = [["Atributo", "Significado", "Tipo / unidade"],
         ["gender", "Sexo biológico (Female, Male, Other)", "categórico"],
         ["age", "Idade (0,08 a 80)", "numérico, anos"],
         ["hypertension", "Diagnóstico de hipertensão", "binário 0/1"],
         ["heart_disease", "Diagnóstico de doença cardíaca", "binário 0/1"],
         ["smoking_history", "Histórico de tabagismo (never, former, current, not current, ever, No Info)", "categórico"],
         ["bmi", "Índice de massa corporal (10,0 a 95,7)", "numérico, kg/m²"],
         ["HbA1c_level", "Hemoglobina glicada, média da glicemia dos últimos ~3 meses (3,5 a 9,0)", "numérico, %"],
         ["blood_glucose_level", "Glicose no sangue (80 a 300)", "numérico, mg/dL"],
         ["<b>diabetes</b>", "<b>Variável-alvo</b>: 1 = tem diabetes, 0 = não tem", "binário 0/1"]]
S += [P("3. Dataset", h1),
      P("<b>Origem:</b> “Diabetes prediction dataset”, publicado no Kaggle por Mohammed Mustafa (arquivo <font name='M'>diabetes_prediction_dataset.csv</font>, "
        "baixado em abril de 2023). São <b>100.000 registros e 8 atributos preditores</b> mais o alvo, sem valores ausentes. A classe positiva representa "
        f"<b>{pct(info['prevalencia_bruta'])}</b> dos registros (desbalanceamento de ~11:1). Escolhemos este dataset por ser público, grande, com variáveis que "
        "um usuário leigo conhece e por permitir comparar modelos lineares e baseados em árvores."),
      tbl(attrs, [3.3, 10.2, 3.5]), Spacer(1, 4),
      P("<b>Limitação importante – dados sintéticos.</b> Durante a análise exploratória observamos que a variável-alvo segue uma regra quase determinística: "
        f"todos os {fmtn(X['zonas']['alta']['n'])} registros com HbA1c ≥ 6,8% <i>ou</i> glicose ≥ 220 mg/dL têm diabetes; todos os {fmtn(X['zonas']['baixa']['n'])} com "
        f"HbA1c ≤ 5,0% <i>ou</i> glicose ≤ 100 mg/dL não têm; e, nos demais {fmtn(X['zonas']['intermediaria']['n'])} registros (zona intermediária), a taxa é ~7% e quase não varia com o valor do exame "
        "(Figura 1, gráfico inferior). Isso não é como a doença real se comporta e é compatível com a descrição do dataset como sintético/derivado em "
        "literatura que o utiliza (Healthcare, 2025). Consequência: as métricas medem a capacidade de recuperar o padrão <i>deste</i> conjunto, "
        "não desempenho clínico.", body),
      fig("eda.png", 15.5), fig("label_structure.png", 13.5),
      P("Figura 1 – Acima: distribuição do alvo e dos exames por classe. Abaixo: % com diabetes por valor de HbA1c e de glicose (barras vermelhas = 100%).", cap)]

# ---- 4 Metodologia
cfg = [["Modelo", "Hiperparâmetros testados (GridSearchCV, 5 folds)", "Escolhido"],
       ["Regressão Logística", "C em {0,1; 1; 10}; class_weight em {nenhum, balanced}; variáveis padronizadas", "C=1, sem class_weight"],
       ["Árvore de Decisão", "max_depth em {2,3,4,5,6,8,12,None}; min_samples_leaf em {1,20}; class_weight em {nenhum, balanced}", "max_depth=8, min_samples_leaf=1, sem class_weight"],
       ["Random Forest (pesquisado além do conteúdo da aula)", "100 árvores; max_depth em {6,10}; min_samples_leaf=5; class_weight em {nenhum, balanced}", "max_depth=10, sem class_weight"]]
S += [P("4. Metodologia", h1), P("<b>[IA]</b> O script <font name='M'>train.py</font> foi escrito com apoio do Claude e revisado pelo grupo.", small), Spacer(1, 3),
      P("Preparação dos dados", h2),
      B(f"<b>Duplicatas:</b> {fmtn(info['duplicatas'])} linhas duplicadas ({pct(info['duplicatas']/info['registros_brutos'], 2)}) removidas <i>antes</i> da divisão, "
        "para que a mesma pessoa não apareça no treino e no teste."),
      B(f"<b>Categoria “Other” em sexo:</b> {info['gender_other']} registros removidos; é pouco para aprender um padrão e a aplicação aceitaria uma opção sem sustentação nos dados."),
      B(f"Resultado: <b>{fmtn(info['registros_limpos'])} registros</b> ({pct(info['prevalencia_limpa'], 2)} positivos)."),
      B("<b>Valores ausentes:</b> não há. “No Info” em tabagismo (36% dos registros) foi mantido como categoria própria, pois é informação ausente codificada e pode ser informativa."),
      B("<b>Codificação:</b> one-hot para sexo e tabagismo (categorias desconhecidas são ignoradas na inferência). <b>Normalização:</b> StandardScaler apenas para a regressão "
        "logística; árvores não dependem de escala. Tudo dentro de um <font name='M'>Pipeline</font>, portanto ajustado só no treino (e só nas dobras de treino durante a validação cruzada)."),
      B("<b>IMC extremo</b> (até 95,7) foi mantido: árvores são robustas a outliers e não há evidência de erro de registro."),
      P("Divisão e procedimento de avaliação", h2),
      P(f"Divisão estratificada 80/20 (semente 42): <b>{fmtn(M['split']['treino'])} treino</b> ({fmtn(M['split']['pos_treino'])} positivos) e <b>{fmtn(M['split']['teste'])} teste</b> "
        f"({fmtn(M['split']['pos_teste'])} positivos)." +
        " A escolha de modelo e hiperparâmetros foi feita só no treino, por validação cruzada estratificada de 5 dobras com <b>F1 da classe diabetes</b> como critério: "
        "com 91% de negativos, a acurácia sozinha é enganosa (um modelo que sempre responde “não” acertaria 91%). O teste foi usado uma única vez, após fixar as configurações."),
      tbl(cfg, [4.3, 8.2, 4.5]), Spacer(1, 3),
      P("<b>Regra de escolha para deploy (definida antes de ver o teste):</b> maior F1 de validação cruzada; se a diferença for menor que 0,005, preferir o modelo mais simples e interpretável.")]

# ---- 5 Resultados
def row(n, key):
    t = R[n]["test"]
    return [n.replace(" (pesquisado além do conteúdo da aula)", ""), f3(R[n]["cv_f1"]), f3(t["precision"]), f3(t["recall"]), f3(t["f1"]), f3(t["f1_neg"]),
            f3(t["accuracy"]), f3(t["roc_auc"]), f3(t["pr_auc"])]
res = [["Modelo", "F1 CV", "Precisão (diab.)", "Recall (diab.)", "F1 (diab.)", "F1 (sem diab.)", "Acurácia", "ROC-AUC", "PR-AUC"]]
for n in ["Regressão Logística", "Árvore de Decisão", "Random Forest"]:
    res.append(row(n, None))
res[2] = [f"<b>{c}</b>" for c in res[2]]
fa = R["analise_erros"]; fx = X["faixas"]; li = X["limiares"]
S += [P("5. Resultados", h1),
      tbl(res, [3.6, 1.4, 1.9, 1.8, 1.6, 1.9, 1.6, 1.6, 1.6]),
      P("Tabela 1 – Validação cruzada (treino) e métricas no conjunto de teste (limiar 0,5). Em negrito, o modelo escolhido para deploy.", cap),
      fig("confusion_matrices.png", 16.5),
      Table([[fig("tree_depth.png", 7.6), fig("importance.png", 7.6)]], colWidths=[8.5 * cm, 8.5 * cm]),
      P("Figura 2 – Matrizes de confusão no teste (acima); efeito da profundidade da árvore no F1 de treino e validação e importância por permutação das variáveis no modelo escolhido (abaixo).", cap),
      P("Comparação e justificativa", h2),
      P(f"A <b>regressão logística</b> ficou abaixo (F1 {f3(R['Regressão Logística']['test']['f1'])}; recall {f3(R['Regressão Logística']['test']['recall'])}) porque o padrão dos dados é em "
        "degraus (limiares de HbA1c e glicose), que um modelo linear aproxima mal, e gera 176 falsos positivos contra 12 da árvore. A <b>árvore</b> e a <b>Random Forest</b> "
        f"empatam na prática (F1 de CV {f3(R['Árvore de Decisão']['cv_f1'])} e {f3(R['Random Forest']['cv_f1'])}; diferença &lt; 0,005). Pela regra pré-definida escolhemos a "
        "<b>Árvore de Decisão (max_depth=8)</b>: é mais simples, rápida, explicável em regras e dispensa pré-processamento de escala. A curva de profundidade mostra "
        "um platô: com 2 níveis o F1 de CV já é 0,799 e acima de ~10 o modelo passa a decorar o treino (profundidade 20: F1 de treino 0,937 contra 0,749 na validação). "
        "Não usamos <i>class_weight</i> no modelo final: ele não melhorou o F1 de CV e distorceria as probabilidades que a aplicação exibe."),
      P("Por que o recall para em ~0,69", h2),
      P(f"Uma regra pura (“HbA1c ≥ 6,8 ou glicose ≥ 220”) já alcança precisão 1,00, recall {f3(X['regra_pura']['recall'])} e F1 {f3(X['regra_pura']['f1'])}, praticamente o "
        f"mesmo que os três modelos. Os {fmtn(X['zonas']['intermediaria']['positivos'])} positivos restantes (33% do total) estão na zona intermediária (HbA1c entre 5,1 e 6,7 e glicose entre 101 e 219), "
        f"onde a taxa de diabetes é {pct(X['zonas']['intermediaria']['taxa'])} e só depende de idade (de 0,8% até 30 anos a 18,1% acima de 70), IMC, hipertensão e doença cardíaca. "
        f"No teste, o modelo errou {fa['falsos_negativos']} falsos negativos (HbA1c média {fa['fn_hba1c_medio']:.1f}%, glicose média {fa['fn_glicose_media']:.0f} mg/dL, idade média {fa['fn_idade_media']:.0f} anos) "
        f"e apenas {fa['falsos_positivos']} falsos positivos. Esses erros são <b>informação que o dataset não contém</b>, não falha de ajuste; nenhum modelo testado os resolve. "
        "A importância por permutação confirma: HbA1c (queda de 0,356 no F1) e glicose (0,273) dominam; IMC, idade e doença cardíaca contribuem ≤ 0,004; sexo, tabagismo e hipertensão, 0."
        ),
      Table([[fig("risk_bands.png", 6.8), fig("roc_pr.png", 10.0)]], colWidths=[7.0 * cm, 10.1 * cm]),
      P("Figura 3 – Faixas de risco: probabilidade prevista versus taxa observada no teste (esq.); curvas ROC e Precisão-Recall (dir.).", cap),
      P(f"<b>Probabilidades confiáveis.</b> As faixas exibidas na aplicação (&lt;10%, 10–50%, ≥50%), fixadas antes dos experimentos, correspondem a taxas reais de "
        f"{pct(fx[0]['taxa_real'])}, {pct(fx[1]['taxa_real'])} e {pct(fx[2]['taxa_real'])} no teste ({pct(fx[0]['pct_teste'],0)}, {pct(fx[1]['pct_teste'],0)} e {pct(fx[2]['pct_teste'],0)} das pessoas), "
        f"ou seja, o modelo está bem calibrado. Como informação descritiva, tratar “moderado ou alto” como alerta elevaria o recall para {pct(li[2]['recall'])} ao custo de precisão de {pct(li[2]['precision'])}: "
        "é um compromisso aceitável para triagem, mas o limiar de classificação do deploy permanece 0,5.")]

# ---- 6 Aplicação
S += [P("6. Aplicação e deploy", h1),
      P("<b>[IA]</b> Código da aplicação (<font name='M'>app.py</font>, <font name='M'>predictor.py</font>, <font name='M'>templates/index.html</font>) gerado com apoio do Claude e revisado pelo grupo.", small), Spacer(1, 3),
      P("A aplicação é um site em <b>Flask</b> com uma página: formulário à esquerda, resultado à direita e limitações sempre visíveis abaixo. Cada consulta executa "
        "<b>uma inferência</b> com o <font name='M'>Pipeline</font> salvo (<font name='M'>models/pipeline.joblib</font>), carregado uma única vez na inicialização; "
        "o pré-processamento (one-hot) faz parte do pipeline, logo a entrada é tratada exatamente como no treino. Os campos mostram unidades (anos, kg/m², %, mg/dL) e "
        "dicas (por exemplo, como calcular o IMC). A validação no servidor rejeita campos vazios, texto no lugar de número, vírgula decimal (aceita e convertida), "
        "valores fora do intervalo plausível (idade 1–100, IMC 10–70, HbA1c 3,5–15, glicose 50–500) e opções inexistentes, com mensagem por campo. Valores válidos mas "
        "fora da faixa vista no treino (por exemplo, idade &lt; 18) geram um aviso de menor confiabilidade. O resultado traz a probabilidade, uma barra com as três faixas, "
        "uma frase de interpretação e as limitações. Há também <font name='M'>POST /api/predict</font> (JSON) e <font name='M'>GET /health</font>."),
      P("<b>Publicação:</b> o repositório inclui <font name='M'>Dockerfile</font> e <font name='M'>Procfile</font> (gunicorn), compatíveis com Render, Hugging Face Spaces (SDK Docker) e "
        "similares. <b>Link:</b> [COLE AQUI O LINK PÚBLICO]. <b>Como usar:</b> (1) abra o link; (2) preencha os oito campos; (3) clique em “Calcular risco” e leia a faixa e a probabilidade. "
        "Para executar localmente: <font name='M'>pip install -r requirements.txt</font> e <font name='M'>python app.py</font> (detalhes no README).")]

# ---- 7 Testes
def short(r):
    d = r["entrada"]
    return (f"{d['age']} anos, {'F' if d['gender']=='Female' else 'M'}, IMC {d['bmi']}, HbA1c {d['HbA1c_level']}, gl. {d['blood_glucose_level']}, "
            f"HAS {d['hypertension'] or '–'}, DC {d['heart_disease'] or '–'}, tab. {d['smoking_history']}").replace(".", ",") if r["id"] != "T4" else \
           "idade “abc”; sexo vazio; IMC 500; HbA1c vazia; doença cardíaca vazia"
obt = {
    "T1": "0,0% – risco baixo", "T2": "100,0% – risco alto", "T3": "40,0% – risco moderado",
    "T4": "HTTP 400, nenhum cálculo; 5 mensagens de erro, uma por campo",
    "T5": "4,5% – risco baixo", "T6": "59,6% – risco alto", "T7": "0,0% – risco baixo, com aviso de idade < 18"}
com = {
    "T1": "Coerente: exames normais e jovem.", "T2": "Coerente: HbA1c ≥ 6,8 e glicose ≥ 220 sempre são diabetes no treino.",
    "T3": "Zona cinzenta: idade, IMC e comorbidades elevam a probabilidade; o modelo não decide, apenas estima.",
    "T4": "Campos destacados em vermelho; valores digitados são mantidos no formulário.",
    "T5": "<b>Erro do modelo</b> (rótulo real = diabetes). Ilustra a limitação da seção 5: o dataset não traz o suficiente para identificá-lo.",
    "T6": "<b>Erro do modelo</b> (rótulo real = sem diabetes). IMC 44,8 + doença cardíaca + 62 anos geram risco alto sem exame alterado.",
    "T7": "Sem rótulo real conhecido. A aplicação avisa; o resultado não deve ser tratado como correto."}
esp = {r["id"]: r["esperado"] for r in T}
tt = [["Caso", "Entrada", "Esperado", "Obtido", "Comentário"]]
for r in T:
    tt.append([f"<b>{r['id']}</b><br/>{r['tipo']}", short(r), esp[r["id"]], obt[r["id"]], com[r["id"]]])
ok = all(r["consistente"] for r in T)
S += [P("7. Testes e limitações", h1),
      P("<b>[IA]</b> O script <font name='M'>tests/run_tests.py</font>, que executa os casos e tira as capturas de tela, foi escrito com apoio do Claude; os casos T5 e T6 foram selecionados do conjunto de teste por terem rótulo real conhecido.", small), Spacer(1, 3),
      tbl(tt, [1.5, 3.7, 3.9, 3.0, 4.9]), Spacer(1, 3),
      P(f"Cada entrada foi executada no código de inferência (<font name='M'>predictor.py</font>) e na aplicação via HTTP; {'as 7 saídas coincidiram (probabilidade idêntica na casa decimal exibida)' if ok else 'HÁ DIVERGÊNCIAS'}. "
        "Capturas de tela completas estão em <font name='M'>casos_de_teste.pdf</font> e <font name='M'>tests/screenshots/</font>. Antes da entrega o grupo deve repetir pelo menos T1–T4 no <b>link público</b>, em outro navegador."),
      P("Limitações", h2),
      B("<b>Dados sintéticos e sem validação clínica</b> (seção 3): resultados não se transferem para pacientes reais."),
      B(f"<b>Recall de ~{pct(R['Árvore de Decisão']['test']['recall'],0)}</b>: cerca de 3 em cada 10 pessoas com diabetes recebem risco baixo ou moderado; “baixo” não descarta a doença."),
      B("<b>Dependência dos exames:</b> sem HbA1c e glicose o modelo não é confiável; com eles, ele essencialmente reproduz critérios laboratoriais já conhecidos."),
      B("<b>Cobertura:</b> não distingue tipos de diabetes; idades até 80 anos; menores de 18 anos são 17% dos registros e a aplicação os sinaliza."),
      B("<b>Probabilidades discretas:</b> a árvore produz poucos valores distintos (18 no teste), então entradas próximas podem receber exatamente o mesmo risco."),
      B("<b>Sem monitoramento nem re-treino</b> e sem armazenamento das consultas (por escolha, para evitar tratar dados de saúde).")]

# ---- 8 Contribuições / IA / Referências
ia = [["Etapa", "Uso de IA (Claude, Anthropic)", "Revisão / decisão do grupo"],
      ["Análise exploratória e descoberta da estrutura dos rótulos", "Gerou o código de exploração e a hipótese sobre dados sintéticos", "[descrever o que o grupo conferiu]"],
      ["train.py e analysis_extra.py", "Escreveu a primeira versão do código de preparação, GridSearchCV, métricas e figuras", "[descrever ajustes, execução e conferência dos números]"],
      ["Aplicação web (app.py, predictor.py, index.html)", "Gerou o código, a validação de entradas e a interface", "[descrever testes e alterações feitas]"],
      ["Testes (tests/run_tests.py, capturas)", "Escreveu o script de testes e a automação das capturas", "[descrever os casos que o grupo escolheu/validou]"],
      ["Relatório, README e texto das limitações", "Redigiu a primeira versão deste texto e do README", "[descrever a revisão e as alterações]"]]
contrib = [["Integrante", "Contribuição objetiva"],
           ["[NOME 1]", "[ex.: análise exploratória, preparação dos dados e treinamento]"],
           ["[NOME 2]", "[ex.: aplicação web, deploy e testes no link público]"],
           ["[NOME 3]", "[ex.: avaliação, relatório e README]"]]
S += [P("8. Contribuições e referências", h1),
      P("Contribuição de cada integrante", h2), tbl(contrib, [3.2, 13.8]), Spacer(1, 4),
      P("Uso de ferramentas de IA", h2),
      P("Foi utilizado o assistente <b>Claude</b> (Anthropic) como apoio em todas as etapas abaixo. O grupo declara que o código foi executado e os resultados foram conferidos "
        "por seus integrantes; as colunas em colchetes devem ser completadas com o que cada integrante de fato fez."),
      tbl(ia, [4.6, 6.2, 6.2])]
S += [PageBreak(), P("Referências", h1)]
refs = [
    "MUSTAFA, M. <i>Diabetes prediction dataset</i>. Kaggle, 2023. https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset",
    "PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830, 2011.",
    "Interpretable Machine Learning Framework for Diabetes Prediction: Integrating SMOTE Balancing with SHAP Explainability for Clinical Decision Support. "
    "<i>Healthcare</i>, 13(20), 2588, 2025. doi:10.3390/healthcare13202588 (descreve o dataset do Kaggle como sintético/derivado e não clínico).",
    "The pandas development team. pandas; HARRIS, C. R. et al. NumPy; HUNTER, J. D. Matplotlib; Pallets Projects. Flask; Gunicorn; ReportLab; Microsoft. Playwright for Python (versões em requirements.txt).",
    "AMERICAN DIABETES ASSOCIATION. <i>Standards of Care in Diabetes</i> (critérios diagnósticos: HbA1c ≥ 6,5% e glicemia de jejum ≥ 126 mg/dL), consulta para contextualização clínica.",
    "ANTHROPIC. <i>Claude</i> (assistente de IA generativa), usado conforme a seção 8.",
    "Material da disciplina Tópicos Avançados em IA (Prof. Dr. Márcio Andrey Teixeira): enunciado do trabalho e conteúdo de aula."]
S += [P(f"[{i+1}] {r}", body) for i, r in enumerate(refs)]
make(os.path.join(ROOT, "relatorio.pdf"), S)

# ===================================================================================== CASOS DE TESTE
S2 = [P("GlicoRisco – Casos de teste", ParagraphStyle("t", parent=ttl, fontSize=18, leading=22)), Spacer(1, 4),
      P("Entradas, comportamento esperado, resultado obtido e capturas de tela da aplicação. A saída do modelo não é tratada como correta quando não há rótulo real conhecido. "
        "Resultados gerados por <font name='M'>tests/run_tests.py</font> (inferência local e requisição HTTP à aplicação, com resultado idêntico em todos os casos). "
        "<b>[IA]</b> Script de teste e capturas automatizadas com apoio do Claude. <b>Substitua as capturas pelas do link público antes da entrega, se desejar.</b>", body), Spacer(1, 6)]
from PIL import Image as PI
for i, r in enumerate(T):
    p = os.path.join(ROOT, "tests", "screenshots_crop", f"{r['id']}.png"); w, h = PI.open(p).size
    wc = 14.2; hc = wc * h / w
    blk = [P(f"{r['id']} – {r['nome']} ({r['tipo']})", h2),
           P(f"<b>Entrada:</b> {short(r)}<br/><b>Esperado:</b> {esp[r['id']]}<br/><b>Obtido:</b> {obt[r['id']]}<br/><b>Comentário:</b> {com[r['id']]}", small),
           Spacer(1, 3), Image(p, width=wc * cm, height=hc * cm), Spacer(1, 8)]
    S2.append(KeepTogether(blk))
make(os.path.join(ROOT, "casos_de_teste.pdf"), [NextPageTemplate("main")] + S2)
print("ok")
