# GlicoRisco – triagem de risco de diabetes com aprendizado de máquina

Trabalho prático da disciplina **Tópicos Avançados em Inteligência Artificial** (Prof. Dr. Márcio Andrey Teixeira).

|**Aplicação publicada**|**https://glicorisco.onrender.com**|
|-|-|
|**Integrantes**|Gabriel Cardoso Cezare, Flávio Gabriel Boar, João Pedro Florenço da Silva.|

A aplicação recebe idade, sexo, IMC, HbA1c, glicose, hipertensão, doença cardíaca e histórico de tabagismo, e devolve a **probabilidade estimada de diabetes**, uma **faixa de risco** (baixo, moderado, alto) e a classificação do modelo. É uma ferramenta **educacional**, não diagnóstica.

## Executar na máquina pessoal

Para executar a aplicação localmente, é necessário ter o **Python 3.12** instalado. Abra um terminal na pasta do projeto e siga as instruções para o seu sistema operacional. Os comandos abaixo criam um ambiente virtual, instalam as dependências e iniciam o servidor Flask.

O pipeline treinado já está incluído em `models/`; **não é necessário executar `train.py`** para usar a aplicação.

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Se o PowerShell bloquear a ativação do ambiente virtual, execute `.\.venv\Scripts\activate.bat` no Prompt de Comando (cmd) em vez de ativá-lo no PowerShell.

Quando o servidor iniciar, abra **http://127.0.0.1:5000** no navegador. Para encerrá-lo, pressione `Ctrl+C` no terminal. A rota **http://127.0.0.1:5000/health** permite verificar se a aplicação está respondendo.

### Alternativa com Docker

Com o Docker instalado e em execução, estes comandos funcionam no Linux, no Windows PowerShell e no Prompt de Comando, a partir da pasta do projeto:

```bash
docker build -t glicorisco .
docker run --rm -p 7860:7860 glicorisco
```

Com o contêiner em execução, acesse **http://127.0.0.1:7860**. Para encerrá-lo, pressione `Ctrl+C`.

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
