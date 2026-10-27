# GlicoRisco – triagem de risco de diabetes com aprendizado de máquina

Trabalho prático da disciplina **Tópicos Avançados em Inteligência Artificial** (Prof. Dr. Márcio Andrey Teixeira).

|**Aplicação publicada**|**https://glicorisco.onrender.com**|
|-|-|
|**Integrantes**|Gabriel Cardoso Cezare, Flávio Gabriel Boar, João Pedro Florenço da Silva.|

A aplicação recebe idade, sexo, IMC, HbA1c, glicose, hipertensão, doença cardíaca e histórico de tabagismo, e devolve a **probabilidade estimada de diabetes**, uma **faixa de risco** (baixo, moderado, alto) e a classificação do modelo. É uma ferramenta **educacional**, não diagnóstica.

## Resumo técnico

* **Problema:** classificação binária supervisionada (`diabetes` = 0/1).
* **Dataset:** [Diabetes prediction dataset (Kaggle, M. Mustafa)](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset), 100.000 × 9. Cópia em `data/`. Após remover 3.854 duplicatas e 18 registros `gender = Other`: 96.128 registros (8,82% positivos).
* **Divisão:** 80/20 estratificada (semente 42). Escolha de modelo por validação cruzada (5 dobras) só no treino; teste usado uma vez.
* **Modelos comparados:** Regressão Logística, **Árvore de Decisão (escolhida, `max\\\_depth=8`)** e Random Forest.
* **Resultado no teste (árvore):** precisão 0,990 · recall 0,685 · F1 0,810 (classe diabetes) · acurácia 0,972 · ROC-AUC 0,971.
* **Achado importante:** o dataset se comporta como **sintético** (HbA1c ≥ 6,8 ou glicose ≥ 220 ⇒ sempre diabetes; HbA1c ≤ 5,0 ou glicose ≤ 100 ⇒ nunca). Por isso o recall tem um teto de \~0,69 para qualquer modelo. Detalhes no relatório.

## Estrutura

```
app.py                  aplicação Flask (formulário, /predict, /api/predict, /health)
predictor.py            validação das entradas + inferência (usado pela app e pelos testes)
templates/index.html    interface
train.py                preparação dos dados, treinamento, comparação e avaliação
analysis\\\_extra.py       estrutura dos rótulos, faixas de risco, limiares (rodar após train.py)
models/                 pipeline.joblib (pré-processamento + modelo) e metadata.json
data/                   diabetes\\\_prediction\\\_dataset.csv
results/                metrics.json, extra.json, train\\\_log.txt
report/                 figuras (figures/) e build\\\_report.py (gera os PDFs)
relatorio.pdf           relatório (capa + 5 páginas + referências)
casos\\\_de\\\_teste.pdf      casos de teste com capturas de tela
tests/                  run\\\_tests.py, results.json, screenshots/
requirements.txt  Dockerfile  Procfile
```

