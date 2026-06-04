# Inspeção Visual Automática de Frutas — *fresh* × *rotten*

Protótipo de **visão computacional clássica** para inspeção de qualidade de frutas
em uma central de distribuição (Cenário A). O sistema separa frutas **OK (fresh)**
de **defeituosas (rotten)** a partir de imagens RGB, usando **features manuais**
(forma, momentos de Hu, cor e textura) e **classificadores clássicos** do
scikit-learn. Inclui um módulo de **explicabilidade (XAI)** como bônus.

## Resultado principal

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC |
|--------|:-:|:-:|:-:|:-:|:-:|
| **SVM (RBF)** | **0.975** | 0.975 | 0.975 | **0.975** | **0.994** |
| KNN | 0.951 | 0.950 | 0.950 | 0.950 | 0.980 |
| Random Forest | 0.938 | 0.927 | 0.950 | 0.938 | 0.983 |
| Regressão Logística | 0.914 | 0.902 | 0.925 | 0.914 | 0.959 |

*(conjunto de teste, 81 imagens; ver `outputs/metricas/tabela_comparativa.csv`)*

As features de **cor** e **textura** dominam a decisão (confirmado por SHAP,
permutation importance e *ablation study*) — coerente com o domínio: a podridão se
manifesta como escurecimento, manchas marrons e textura irregular.

## Dataset

*Fruits fresh and rotten for classification* (Kaggle) — maçã, banana e laranja,
fundo controlado. Estrutura esperada:

```
dataset/train/{freshapples,freshbanana,freshoranges,rottenapples,rottenbanana,rottenoranges}/
```

Usamos **somente as imagens originais** (as começadas por `Screen Shot`),
descartando as versões aumentadas (`rotated_by_*`, `translation_*`,
`saltandpepper_*`, `vertical_flip_*`) — elas são derivadas das mesmas frutas e
causariam **vazamento de dados** entre treino e teste. Amostramos **~200 imagens
por classe**, balanceadas entre as 3 frutas (67 por fruta×condição).

## Como reproduzir (3 comandos)

```bash
pip install -r requirements.txt
python -m src.build_dataset
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

1. instala as dependências;
2. roda o pipeline (amostragem → segmentação → extração de features) e gera
   `outputs/X.csv` e `outputs/y.csv`;
3. executa os 4 notebooks, regenerando todas as figuras, matrizes de confusão e
   tabelas de métricas em `outputs/`.

## Interface Streamlit (bônus)

App interativo com abas para visão geral do dataset, comparação de segmentação,
**treino e comparação dos modelos**, métricas por modelo, **XAI** (importâncias,
permutation, coeficientes, ablation, SHAP) e **predição** de uma foto enviada.

```bash
streamlit run app.py
```

## Pipeline

```
Aquisição (RGB)  ->  Pré-processamento  ->  Segmentação  ->  Extração de features  ->  Classificação  ->  Métricas
  amostragem        blur/realce/HSV       Otsu vs HSV       forma+Hu+cor+textura      KNN/LogReg/RF/SVM   acc/prec/rec/F1/ROC
  balanceada                              (escolhido HSV)   + indicadores de podridão  + GridSearchCV      + matriz de confusão
```

## Estrutura do repositório

```
PipelineFrutas/
├── src/                      # código do pipeline (importável e reutilizável)
│   ├── config.py             # caminhos, classes, amostragem, random_state
│   ├── dataset.py            # amostragem reprodutível (só originais) + carregamento
│   ├── segmentation.py       # 2 métodos: Otsu e HSV (+ pós-processamento)
│   ├── features.py           # forma, Hu, cor, textura (GLCM/LBP), podridão
│   ├── modeling.py           # split treino/val/teste + fábrica de modelos
│   ├── plots.py              # matrizes de confusão, ROC, boxplots
│   ├── build_dataset.py      # gera X.csv e y.csv
│   ├── viz_segmentation.py   # figuras de comparação de segmentação
│   └── make_notebooks.py     # gera os notebooks programaticamente
├── notebooks/
│   ├── 01_segmentacao.ipynb
│   ├── 02_features.ipynb
│   ├── 03_classificacao.ipynb
│   └── 04_xai_bonus.ipynb
├── app.py                    # interface Streamlit (treino/comparação/XAI/predição)
├── src/pipeline_api.py       # API de treino e predição usada pelo app
├── outputs/                  # X.csv, y.csv, figuras, matrizes, métricas, erros
├── relatorio/                # relatório técnico (Markdown -> PDF)
├── requirements.txt
└── README.md
```

## Reprodutibilidade

`random_state = 42` é fixado em amostragem, splits, validação cruzada e
classificadores. O `StandardScaler` é ajustado **dentro do `Pipeline`** (somente no
treino de cada fold), evitando vazamento de dados.

## Bibliotecas

numpy · pandas · opencv-python · scikit-image · scikit-learn · scipy · matplotlib
· seaborn · shap. Trechos de referência sobre XAI seguem Molnar, *Interpretable
Machine Learning* (christophm.github.io/interpretable-ml-book).
