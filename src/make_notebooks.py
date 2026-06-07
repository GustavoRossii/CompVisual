from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

PREAMBLE = (
    "import sys, os\n"
    "from pathlib import Path\n"
    "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
    "sys.path.insert(0, str(ROOT))\n"
    "import warnings; warnings.filterwarnings('ignore')\n"
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
    "sns.set_theme(style='whitegrid')\n"
    "pd.set_option('display.width', 120); pd.set_option('display.max_columns', 60)\n"
)


def build(cells):
    nb = new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    return nb


# ========================================================================== #
# 01 - SEGMENTACAO                                                            #
# ========================================================================== #
def nb_segmentacao():
    c = []
    c.append(new_markdown_cell(
        "# 01 — Aquisição, pré-processamento e **segmentação**\n\n"
        "**Projeto:** Inspeção visual automática de frutas — Cenário A (central de "
        "distribuição).\n\n"
        "**Problema.** Separar frutas **OK (fresh)** de **defeituosas (rotten)** em alta "
        "velocidade, substituindo a inspeção humana (cara, lenta, inconsistente e "
        "subjetiva).\n\n"
        "**Dataset.** *Fruits fresh and rotten for classification* (Kaggle) — maçã, banana "
        "e laranja, fundo controlado. Usamos **somente as imagens originais** (descartamos "
        "as versões aumentadas por rotação/translação/ruído presentes no dataset, pois são "
        "derivadas das mesmas frutas e causariam **vazamento de dados** entre treino e "
        "teste).\n\n"
        "**Tarefa de classificação:** binária **fresh × rotten**, com as 3 frutas juntas e "
        "**balanceadas** (~200 imagens por classe → 67 por fruta×condição).\n\n"
        "Este notebook cobre a **aquisição**, o **pré-processamento** e a **segmentação** "
        "(isolamento do objeto), comparando **dois métodos**."
    ))
    c.append(new_code_cell(
        "print('Total de imagens amostradas:', len(index))\n"
        "display(index.groupby(['fruit','label_name']).size().unstack())"
    ))
    c.append(new_markdown_cell(
        "## Exemplos por classe\n"
        "Frutas isoladas sobre fundo claro. Note os sinais visuais de podridão: "
        "manchas escuras, escurecimento (marrom), rugas e textura irregular."
    ))
    c.append(new_code_cell(
        "fig, axes = plt.subplots(2, 3, figsize=(9, 6))\n"
        "for ax, folder in zip(axes.ravel(), cfg.FOLDER_INFO):\n"
        "    r = index[index.folder==folder].iloc[0]\n"
        "    ax.imshow(ds.load_image(r['path'])); ax.set_title(folder); ax.axis('off')\n"
        "fig.suptitle('Exemplos do dataset (originais)'); fig.tight_layout()\n"
        "fig.savefig(cfg.FIG_DIR/'exemplos_dataset.png', dpi=130); plt.show()"
    ))
    c.append(new_markdown_cell(
        "## Dois métodos de segmentação\n\n"
        "**Método A — Otsu (tons de cinza).** Limiar global automático sobre a imagem em "
        "tons de cinza; o fundo claro é separado do objeto mais escuro.\n\n"
        "**Método B — Cor em HSV (remoção de fundo).** O fundo (branco nas originais, preto "
        "nas aumentadas) tem **baixa saturação** e valor (V) muito alto **ou** muito baixo; "
        "a fruta é colorida. Definimos objeto = *não fundo*.\n\n"
        "Ambos passam por limpeza morfológica, **preenchimento de buracos** e seleção do "
        "**maior componente conectado**."
    ))
    c.append(new_code_cell(
        "from src import viz_segmentation as vz\n"
        "out = vz.comparison_grid(index)\n"
        "from IPython.display import Image; Image(str(out))"
    ))
    c.append(new_markdown_cell(
        "## Comparação quantitativa: cobertura da máscara\n"
        "A **cobertura** é a fração da imagem ocupada pela máscara. Comparamos a "
        "estabilidade dos dois métodos em toda a amostra."
    ))
    c.append(new_code_cell(
        "import numpy as np\n"
        "rows=[]\n"
        "for _,r in index.iterrows():\n"
        "    img = ds.load_image(r['path'])\n"
        "    rows.append({'otsu': seg.mask_coverage(seg.segment_otsu(img)),\n"
        "                 'hsv':  seg.mask_coverage(seg.segment_hsv(img)),\n"
        "                 'folder': r['folder']})\n"
        "cov = pd.DataFrame(rows)\n"
        "print('Cobertura média / desvio por método:')\n"
        "display(cov[['otsu','hsv']].agg(['mean','std','min','max']).round(3))\n"
        "# falhas grosseiras: mascara ~vazia ou ~imagem inteira\n"
        "for m in ['otsu','hsv']:\n"
        "    falhas = ((cov[m]<0.10)|(cov[m]>0.99)).sum()\n"
        "    print(f'{m}: {falhas} casos extremos (cov<0.10 ou >0.99)')"
    ))
    c.append(new_code_cell(
        "fig,ax=plt.subplots(figsize=(6,3.5))\n"
        "sns.kdeplot(cov['otsu'],label='Otsu',fill=True,ax=ax)\n"
        "sns.kdeplot(cov['hsv'],label='HSV',fill=True,ax=ax)\n"
        "ax.set_title('Distribuição da cobertura da máscara'); ax.set_xlabel('cobertura'); ax.legend()\n"
        "fig.tight_layout(); fig.savefig(cfg.SEG_DIR/'cobertura_otsu_vs_hsv.png',dpi=130); plt.show()"
    ))
    c.append(new_markdown_cell(
        "## Discussão e escolha\n\n"
        "- **Otsu** falha quando a fruta é clara e próxima do fundo (ex.: **laranjas "
        "frescas**), chegando a segmentar a imagem inteira ou quase nada — o limiar único "
        "de cinza não separa bem objetos claros sobre fundo claro.\n"
        "- **HSV** é mais robusto porque usa a baixa saturação do fundo, lidando tanto com "
        "fundo branco quanto com o preto das bordas de imagens rotacionadas.\n"
        "- Casos de cobertura ~1.0 no HSV correspondem a **recortes justos** (a fruta "
        "preenche o quadro), não a falhas.\n\n"
        "**Escolhemos o método HSV** para o pipeline principal. Segmentação imperfeita é "
        "parte do desafio real e é tratada com o pós-processamento morfológico."
    ))
    return build(c)


# ========================================================================== #
# 02 - FEATURES                                                               #
# ========================================================================== #
def nb_features():
    c = []
    c.append(new_markdown_cell(
        "# 02 — Extração, análise e seleção de **features**\n\n"
        "Para cada fruta segmentada extraímos um vetor de **features manuais** cobrindo as "
        "famílias pedidas (Aula 8): **forma**, **momentos de Hu (inerciais)**, **cor** e "
        "**textura**, além de **indicadores de podridão** (manchas escuras/marrons). "
        "Todas as estatísticas de cor/textura usam **apenas os pixels da máscara**."
    ))
    c.append(new_code_cell(PREAMBLE +
        "from src import config as cfg, dataset as ds, segmentation as seg, features as ft\n"
        "from src import build_dataset\n"
        "# (Re)gera X.csv e y.csv de forma reprodutível\n"
        "build_dataset.main()"
    ))
    c.append(new_code_cell(
        "X = pd.read_csv(cfg.OUTPUTS/'X.csv'); y = pd.read_csv(cfg.OUTPUTS/'y.csv')\n"
        "feat = ft.all_feature_names()\n"
        "print('X:', X.shape, '| nº de features:', len(feat))\n"
        "print('\\nFamílias de features:')\n"
        "for g,cols in ft.FEATURE_GROUPS.items(): print(f'  {g:9s} ({len(cols)}): '+', '.join(cols))\n"
        "display(X[feat].describe().T.head(12))"
    ))
    c.append(new_markdown_cell("## Coerência das features com o problema\n"
        "Em *fresh × rotten*, esperamos que **cor** (escurecimento, manchas marrons) e "
        "**textura** (rugosidade, irregularidade da casca) separem melhor que a **forma**, "
        "já que a podridão altera principalmente a superfície."
    ))
    c.append(new_markdown_cell("## Boxplots por classe (features candidatas mais relevantes)"))
    c.append(new_code_cell(
        "from src import plots\n"
        "destaque = ['dark_ratio','brown_ratio','sat_std','mean_S',\n"
        "            'glcm_contrast','glcm_homogeneity','glcm_energy','glcm_correlation',\n"
        "            'mean_V','std_V','circularity','solidity']\n"
        "plots.boxplots_by_class(X[feat], y['label'].values, destaque,\n"
        "                        save_as='boxplots_features.png'); plt.show()"
    ))
    c.append(new_markdown_cell("## Médias por classe e *effect size* (quais features separam melhor)"))
    c.append(new_code_cell(
        "d = X[feat].copy(); d['label_name']=y['label_name']\n"
        "means = d.groupby('label_name')[feat].mean().T\n"
        "# Cohen's d como medida de separação\n"
        "def cohend(a,b):\n"
        "    na,nb=len(a),len(b); sp=np.sqrt(((na-1)*a.std()**2+(nb-1)*b.std()**2)/(na+nb-2))\n"
        "    return (b.mean()-a.mean())/ (sp+1e-9)\n"
        "fresh=X[feat][y.label==0]; rotten=X[feat][y.label==1]\n"
        "means['cohen_d']=[cohend(fresh[f],rotten[f]) for f in feat]\n"
        "means['abs_d']=means['cohen_d'].abs()\n"
        "print('Top 12 features por poder de separação (|Cohen d|):')\n"
        "display(means.sort_values('abs_d',ascending=False).head(12).round(3))"
    ))
    c.append(new_code_cell(
        "top = means.sort_values('abs_d',ascending=False).head(15)\n"
        "fig,ax=plt.subplots(figsize=(6,5))\n"
        "colors=['#b3593c' if v>0 else '#69b34c' for v in top['cohen_d']]\n"
        "ax.barh(top.index[::-1], top['cohen_d'][::-1], color=colors[::-1])\n"
        "ax.set_title(\"Separação fresh×rotten por feature (Cohen's d)\")\n"
        "ax.set_xlabel(\"Cohen's d  (>0 maior em rotten)\")\n"
        "fig.tight_layout(); fig.savefig(cfg.FEAT_DIR/'cohend_features.png',dpi=130); plt.show()"
    ))
    c.append(new_markdown_cell("## Correlação entre features"))
    c.append(new_code_cell(
        "corr = X[feat].corr()\n"
        "fig,ax=plt.subplots(figsize=(11,9))\n"
        "sns.heatmap(corr,cmap='coolwarm',center=0,square=True,\n"
        "            xticklabels=True,yticklabels=True,cbar_kws={'shrink':.6},ax=ax)\n"
        "ax.set_title('Matriz de correlação das features'); plt.xticks(fontsize=6); plt.yticks(fontsize=6)\n"
        "fig.tight_layout(); fig.savefig(cfg.FEAT_DIR/'correlacao.png',dpi=130); plt.show()"
    ))
    c.append(new_markdown_cell("## PCA (visualização 2D) e comparação entre grupos de features"))
    c.append(new_code_cell(
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.decomposition import PCA\n"
        "Z = StandardScaler().fit_transform(X[feat])\n"
        "pca = PCA(n_components=2, random_state=cfg.RANDOM_STATE); P = pca.fit_transform(Z)\n"
        "fig,ax=plt.subplots(figsize=(6,5))\n"
        "for lab,co,nm in [(0,'#69b34c','fresh'),(1,'#b3593c','rotten')]:\n"
        "    s=y.label==lab; ax.scatter(P[s,0],P[s,1],c=co,label=nm,alpha=.6,s=18)\n"
        "ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)')\n"
        "ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)')\n"
        "ax.set_title('PCA das features'); ax.legend()\n"
        "fig.tight_layout(); fig.savefig(cfg.FEAT_DIR/'pca.png',dpi=130); plt.show()"
    ))
    c.append(new_code_cell(
        "# Comparação entre grupos de features (CV rápida com Random Forest)\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.model_selection import cross_val_score, StratifiedKFold\n"
        "cvk = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)\n"
        "combos = {'cor':['cor'],'textura':['textura'],'forma':['forma'],'hu':['hu'],\n"
        "          'podridão':['podridão'],'cor+textura':['cor','textura'],\n"
        "          'cor+textura+forma':['cor','textura','forma'],\n"
        "          'TODAS':list(ft.FEATURE_GROUPS)}\n"
        "res=[]\n"
        "for nm,gs in combos.items():\n"
        "    cols=[c for g in gs for c in ft.FEATURE_GROUPS[g]]\n"
        "    sc=cross_val_score(RandomForestClassifier(300,random_state=cfg.RANDOM_STATE),\n"
        "                       X[cols],y.label,cv=cvk,scoring='f1')\n"
        "    res.append({'grupo':nm,'n_feats':len(cols),'F1_cv':sc.mean(),'std':sc.std()})\n"
        "grp=pd.DataFrame(res).sort_values('F1_cv',ascending=False)\n"
        "grp.to_csv(cfg.METRIC_DIR/'comparacao_grupos_features.csv',index=False)\n"
        "display(grp.round(3))"
    ))
    c.append(new_markdown_cell(
        "## Conclusões da análise de features\n\n"
        "- As features de **cor** e **podridão** (`dark_ratio`, `brown_ratio`, `mean_S`, "
        "`mean_V`) e de **textura** (GLCM `contrast`/`homogeneity`) têm os maiores "
        "*effect sizes* — coerente com o problema, pois a podridão muda cor e textura.\n"
        "- A **forma** e os **momentos de Hu** separam pouco (frutas frescas e podres têm "
        "contornos parecidos), mas mantemos uma família de cada tipo conforme o enunciado.\n"
        "- `cor+textura` já alcança quase o desempenho de *todas* as features.\n\n"
        "Os arquivos **`outputs/X.csv`** e **`outputs/y.csv`** ficam prontos para o notebook 03."
    ))
    return build(c)


# ========================================================================== #
# 03 - CLASSIFICACAO                                                          #
# ========================================================================== #
def nb_classificacao():
    c = []
    c.append(new_markdown_cell(
        "# 03 — Classificação e avaliação\n\n"
        "Treinamos e comparamos **quatro classificadores clássicos** (KNN, Regressão "
        "Logística, Random Forest, SVM) sobre o vetor de features manuais.\n\n"
        "**Metodologia de avaliação:**\n"
        "- Split **estratificado** treino/validação/teste = 60/20/20.\n"
        "- **`StandardScaler` dentro do `Pipeline`** → ajustado só no treino em cada fold "
        "(sem vazamento).\n"
        "- Ajuste de hiperparâmetros com **`GridSearchCV`** (validação cruzada no "
        "treino+validação).\n"
        "- Teste reservado **apenas** para a avaliação final.\n"
        "- Métricas: acurácia, precisão, recall, F1, matriz de confusão e **curva ROC**.\n"
        "- `random_state` fixo em todos os passos."
    ))
    c.append(new_code_cell(PREAMBLE +
        "from src import config as cfg, modeling as mdl, plots\n"
        "X, y, meta = mdl.load_Xy()\n"
        "sp = mdl.make_split(X, y)\n"
        "print('treino/val/teste =', len(sp.ytr), len(sp.yval), len(sp.yte))\n"
        "for nm,arr in [('treino',sp.ytr),('val',sp.yval),('teste',sp.yte)]:\n"
        "    print(f'  {nm}: fresh={int((arr==0).sum())} rotten={int((arr==1).sum())}')"
    ))
    c.append(new_markdown_cell(
        "## Ajuste de hiperparâmetros (GridSearchCV) e validação cruzada\n"
        "Buscamos os hiperparâmetros em **treino+validação** com CV estratificada de 5 folds."
    ))
    c.append(new_code_cell(
        "from sklearn.model_selection import GridSearchCV, StratifiedKFold\n"
        "cvk = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)\n"
        "fitted, cv_rows = {}, []\n"
        "for name,(pipe,grid) in mdl.make_models().items():\n"
        "    gs = GridSearchCV(pipe, grid, scoring='f1', cv=cvk, n_jobs=-1)\n"
        "    gs.fit(sp.Xtrval, sp.ytrval)\n"
        "    fitted[name] = gs.best_estimator_\n"
        "    cv_rows.append({'modelo':name,'F1_cv':gs.best_score_,'melhores_params':gs.best_params_})\n"
        "    print(f'{name:13s} F1_cv={gs.best_score_:.3f}  {gs.best_params_}')\n"
        "pd.DataFrame(cv_rows).to_csv(cfg.METRIC_DIR/'cv_hiperparametros.csv',index=False)"
    ))
    c.append(new_markdown_cell("## Avaliação final no conjunto de teste"))
    c.append(new_code_cell(
        "from sklearn.metrics import (accuracy_score, precision_score, recall_score,\n"
        "    f1_score, roc_auc_score, classification_report)\n"
        "rows=[]\n"
        "for name,model in fitted.items():\n"
        "    yp = model.predict(sp.Xte)\n"
        "    proba = model.predict_proba(sp.Xte)[:,1]\n"
        "    rows.append({'modelo':name,\n"
        "        'acuracia':accuracy_score(sp.yte,yp),\n"
        "        'precisao':precision_score(sp.yte,yp),\n"
        "        'recall':recall_score(sp.yte,yp),\n"
        "        'F1':f1_score(sp.yte,yp),\n"
        "        'ROC_AUC':roc_auc_score(sp.yte,proba)})\n"
        "tabela = pd.DataFrame(rows).sort_values('F1',ascending=False).reset_index(drop=True)\n"
        "tabela.to_csv(cfg.METRIC_DIR/'tabela_comparativa.csv',index=False)\n"
        "display(tabela.round(3).style.background_gradient(cmap='Greens',subset=['F1','ROC_AUC']))"
    ))
    c.append(new_code_cell(
        "best_name = tabela.iloc[0]['modelo']; best = fitted[best_name]\n"
        "print('Melhor modelo:', best_name, '\\n')\n"
        "print(classification_report(sp.yte, best.predict(sp.Xte), target_names=['fresh','rotten']))"
    ))
    c.append(new_markdown_cell("## Matrizes de confusão e curvas ROC"))
    c.append(new_code_cell(
        "for name,model in fitted.items():\n"
        "    plots.plot_confusion(sp.yte, model.predict(sp.Xte),\n"
        "        f'Matriz de confusão — {name}', save_as=f'cm_{name}.png'); plt.show()\n"
        "plots.plot_confusion(sp.yte, best.predict(sp.Xte),\n"
        "    f'Matriz normalizada — {best_name}', save_as=f'cm_{best_name}_norm.png',\n"
        "    normalize='true'); plt.show()"
    ))
    c.append(new_code_cell(
        "plots.plot_roc(fitted, sp.Xte, sp.yte, save_as='roc_todos.png'); plt.show()"
    ))
    c.append(new_markdown_cell("## Análise de erros\n"
        "Mostramos imagens em que o melhor modelo errou, com hipóteses sobre a causa."
    ))
    c.append(new_code_cell(
        "from src import dataset as ds\n"
        "yp_best = best.predict(sp.Xte)\n"
        "err_pos = np.where(yp_best != sp.yte)[0]\n"
        "err_global = sp.idx_te[err_pos]\n"
        "print(f'{len(err_global)} erros em {len(sp.yte)} imagens de teste')\n"
        "n = min(8, len(err_global))\n"
        "if n>0:\n"
        "    fig,axes=plt.subplots(2,4,figsize=(12,6)); axes=axes.ravel()\n"
        "    for ax in axes: ax.axis('off')\n"
        "    for k in range(n):\n"
        "        gi=err_global[k]; r=meta.iloc[gi]\n"
        "        true=int(y[gi]); pred=int(yp_best[err_pos[k]])\n"
        "        ax=axes[k]; ax.imshow(ds.load_image(r['path'])); ax.axis('off')\n"
        "        ax.set_title(f\"real={'rotten' if true else 'fresh'}\\npred={'rotten' if pred else 'fresh'}\",fontsize=9,\n"
        "                     color='red')\n"
        "    fig.suptitle(f'Erros do melhor modelo ({best_name})')\n"
        "    fig.tight_layout(); fig.savefig(cfg.ERR_DIR/'erros_melhor_modelo.png',dpi=130); plt.show()\n"
        "    display(meta.iloc[err_global].assign(\n"
        "        real=[ 'rotten' if y[g] else 'fresh' for g in err_global],\n"
        "        pred=[ 'rotten' if p else 'fresh' for p in yp_best[err_pos]])[['fruit','folder','real','pred']])"
    ))
    c.append(new_markdown_cell(
        "## Discussão dos resultados\n\n"
        "- Todos os modelos clássicos atingem **F1 alto** com features manuais simples, "
        "confirmando que cor+textura capturam bem a diferença fresh×rotten.\n"
        "- A **matriz de confusão** e a **ROC** mostram onde cada modelo erra. Os erros "
        "concentram-se em frutas **levemente** maduras (transição fresh→rotten), em "
        "**laranjas** (cuja casca rugosa confunde a textura) e em recortes com segmentação "
        "imperfeita.\n"
        "- A escolha do modelo para produção é discutida na conclusão do relatório "
        "(equilíbrio entre F1, recall de *rotten* — para não deixar passar fruta podre — e "
        "custo/interpretabilidade)."
    ))
    return build(c)


# ========================================================================== #
# 04 - XAI (BONUS)                                                            #
# ========================================================================== #
def nb_xai():
    c = []
    c.append(new_markdown_cell(
        "# 04 — Explicabilidade do modelo (**XAI**) — *bônus / nível avançado*\n\n"
        "Respondemos: **quais features mais influenciaram a decisão do modelo?** Uma "
        "explicação coerente para *fresh × rotten* deve apontar para indicadores de "
        "podridão (manchas escuras/marrons, textura irregular), e **não** para artefatos "
        "do fundo — o que indicaria viés do dataset.\n\n"
        "Métodos aplicados sobre a tabela X de features manuais:\n"
        "1. Coeficientes da Regressão Logística\n"
        "2. Importância de variáveis do Random Forest\n"
        "3. **Permutation importance**\n"
        "4. **SHAP** (TreeExplainer no Random Forest)\n"
        "5. **Ablation study** por grupos de features com validação cruzada"
    ))
    c.append(new_code_cell(PREAMBLE +
        "from src import config as cfg, modeling as mdl, features as ft\n"
        "X, y, meta = mdl.load_Xy(); sp = mdl.make_split(X, y)\n"
        "models = mdl.make_models()\n"
        "from sklearn.model_selection import GridSearchCV, StratifiedKFold\n"
        "cvk = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)\n"
        "def fit_best(name):\n"
        "    pipe,grid = models[name]\n"
        "    gs = GridSearchCV(pipe,grid,scoring='f1',cv=cvk,n_jobs=-1).fit(sp.Xtrval,sp.ytrval)\n"
        "    return gs.best_estimator_\n"
        "rf = fit_best('RandomForest'); logreg = fit_best('LogReg')\n"
        "feat = ft.all_feature_names()\n"
        "print('Modelos ajustados para XAI: RandomForest e LogReg')"
    ))
    c.append(new_markdown_cell("## 1) Coeficientes da Regressão Logística\n"
        "Sinal indica a direção (positivo → empurra para *rotten*)."))
    c.append(new_code_cell(
        "coef = pd.Series(logreg.named_steps['clf'].coef_[0], index=feat).sort_values()\n"
        "top = pd.concat([coef.head(10), coef.tail(10)])\n"
        "fig,ax=plt.subplots(figsize=(6,6))\n"
        "ax.barh(top.index, top.values, color=['#69b34c' if v<0 else '#b3593c' for v in top.values])\n"
        "ax.set_title('Coeficientes da Regressão Logística (features padronizadas)')\n"
        "ax.set_xlabel('coef  (>0 → rotten)')\n"
        "fig.tight_layout(); fig.savefig(cfg.FIG_DIR/'xai_logreg_coef.png',dpi=130); plt.show()"
    ))
    c.append(new_markdown_cell("## 2) Importância de variáveis — Random Forest"))
    c.append(new_code_cell(
        "imp = pd.Series(rf.named_steps['clf'].feature_importances_, index=feat).sort_values(ascending=False)\n"
        "fig,ax=plt.subplots(figsize=(6,5))\n"
        "imp.head(15)[::-1].plot.barh(ax=ax,color='#3c78b3')\n"
        "ax.set_title('Random Forest — importância (Gini) das 15 principais')\n"
        "fig.tight_layout(); fig.savefig(cfg.FIG_DIR/'xai_rf_importance.png',dpi=130); plt.show()\n"
        "display(imp.head(10).round(4))"
    ))
    c.append(new_markdown_cell("## 3) Permutation importance (no conjunto de teste)\n"
        "Mede a queda de F1 quando cada feature é embaralhada — menos enviesada que o Gini."))
    c.append(new_code_cell(
        "from sklearn.inspection import permutation_importance\n"
        "pi = permutation_importance(rf, sp.Xte, sp.yte, scoring='f1',\n"
        "                            n_repeats=20, random_state=cfg.RANDOM_STATE, n_jobs=-1)\n"
        "pis = pd.Series(pi.importances_mean, index=feat).sort_values(ascending=False)\n"
        "fig,ax=plt.subplots(figsize=(6,5))\n"
        "pis.head(15)[::-1].plot.barh(ax=ax,color='#8e44ad')\n"
        "ax.set_title('Permutation importance (queda de F1) — top 15')\n"
        "fig.tight_layout(); fig.savefig(cfg.FIG_DIR/'xai_permutation.png',dpi=130); plt.show()\n"
        "display(pis.head(10).round(4))"
    ))
    c.append(new_markdown_cell("## 4) SHAP — TreeExplainer no Random Forest"))
    c.append(new_code_cell(
        "import shap\n"
        "rf_clf = rf.named_steps['clf']\n"
        "Xte_sc = rf.named_steps['scaler'].transform(sp.Xte)\n"
        "explainer = shap.TreeExplainer(rf_clf)\n"
        "sv = explainer.shap_values(Xte_sc)\n"
        "# para classificador binário, usa contribuições da classe 'rotten'\n"
        "sv_rotten = sv[...,1] if isinstance(sv,np.ndarray) and sv.ndim==3 else (sv[1] if isinstance(sv,list) else sv)\n"
        "shap.summary_plot(sv_rotten, sp.Xte, feature_names=feat, show=False, max_display=15)\n"
        "plt.tight_layout(); plt.savefig(cfg.FIG_DIR/'xai_shap_summary.png',dpi=130,bbox_inches='tight'); plt.show()"
    ))
    c.append(new_markdown_cell("## 5) Ablation study por grupos de features (CV)\n"
        "Treinamos retirando cada grupo para medir seu impacto."))
    c.append(new_code_cell(
        "from sklearn.model_selection import cross_val_score\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "groups = ft.FEATURE_GROUPS\n"
        "base = cross_val_score(RandomForestClassifier(300,random_state=cfg.RANDOM_STATE),\n"
        "                       X[feat], y, cv=cvk, scoring='f1').mean()\n"
        "rows=[{'config':'TODAS','F1_cv':base,'delta':0.0}]\n"
        "for g in groups:\n"
        "    cols=[c for c in feat if c not in groups[g]]\n"
        "    sc=cross_val_score(RandomForestClassifier(300,random_state=cfg.RANDOM_STATE),\n"
        "                       X[cols], y, cv=cvk, scoring='f1').mean()\n"
        "    rows.append({'config':f'sem {g}','F1_cv':sc,'delta':sc-base})\n"
        "abl=pd.DataFrame(rows).sort_values('delta')\n"
        "abl.to_csv(cfg.METRIC_DIR/'ablation_grupos.csv',index=False)\n"
        "display(abl.round(4))\n"
        "fig,ax=plt.subplots(figsize=(6,3.5))\n"
        "sub=abl[abl.config!='TODAS']\n"
        "ax.barh(sub.config, sub.delta, color=['#b3593c' if v<0 else '#69b34c' for v in sub.delta])\n"
        "ax.axvline(0,color='k',lw=.8); ax.set_title('Ablation: variação de F1 ao remover cada grupo')\n"
        "ax.set_xlabel('Δ F1 (negativo = grupo é importante)')\n"
        "fig.tight_layout(); fig.savefig(cfg.FIG_DIR/'xai_ablation.png',dpi=130); plt.show()"
    ))
    c.append(new_markdown_cell(
        "## Interpretação e viés do dataset\n\n"
        "- Os quatro métodos **concordam**: as features mais influentes são de **cor/podridão** "
        "(`brown_ratio`, `dark_ratio`, `mean_S`, `mean_V`) e de **textura** (GLCM "
        "`contrast`/`homogeneity`). Isso é **coerente com o domínio**: a podridão se manifesta "
        "como escurecimento, manchas marrons e textura irregular.\n"
        "- O **ablation** confirma: remover **cor** e **textura** derruba o F1; remover "
        "**forma**/**Hu** quase não afeta.\n"
        "- O modelo **não** depende de forma do contorno nem de elementos do fundo, o que "
        "**reduz o risco de viés** de fundo. Limitação: o dataset tem fundo muito controlado; "
        "em produção, variações de iluminação e fundo exigiriam recalibração das features de cor."
    ))
    return build(c)


def main():
    nbf.write(nb_segmentacao(), NB_DIR / "01_segmentacao.ipynb")
    nbf.write(nb_features(),    NB_DIR / "02_features.ipynb")
    nbf.write(nb_classificacao(), NB_DIR / "03_classificacao.ipynb")
    nbf.write(nb_xai(),         NB_DIR / "04_xai_bonus.ipynb")
    print("Notebooks gerados em", NB_DIR)


if __name__ == "__main__":
    main()
