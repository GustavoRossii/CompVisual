"""
Execução:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config as cfg          
from src import dataset as ds          
from src import features as ft         
from src import segmentation as seg    
from src import pipeline_api as api    

st.set_page_config(page_title="Inspeção de Frutas — fresh × rotten",
                   page_icon="🍎", layout="wide")

GREEN, RED = "#69b34c", "#b3593c"


# --------------------------------------------------------------------------- #
# Helpers com cache                                                            #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_index():
    return ds.build_sample_index()


@st.cache_data(show_spinner=False)
def get_metadata():
    X = pd.read_csv(cfg.OUTPUTS / "X.csv")
    y = pd.read_csv(cfg.OUTPUTS / "y.csv")
    meta = X[["fruit", "folder", "path"]].copy()
    meta["label"] = y["label"].values
    meta["label_name"] = y["label_name"].values
    return meta


@st.cache_data(show_spinner=False)
def data_exists():
    return (cfg.OUTPUTS / "X.csv").exists() and (cfg.OUTPUTS / "y.csv").exists()


@st.cache_data(show_spinner=False)
def dataset_available():
    return cfg.DATASET_DIR.exists()


@st.cache_data(show_spinner="Construindo X.csv / y.csv (segmentação + features)…")
def build_data():
    from src import build_dataset
    build_dataset.main()
    return True


@st.cache_resource(show_spinner="Treinando os 4 modelos (GridSearchCV)…")
def train_models():
    return api.train_all_models()


def read_image_file(uploaded) -> np.ndarray:
    import cv2
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Cabeçalho                                                                     #
# --------------------------------------------------------------------------- #
st.title("🍎 Inspeção Visual Automática de Frutas")
st.caption("Cenário A · classificação **fresh × rotten** com visão computacional "
           "clássica (features manuais + classificadores do scikit-learn) + XAI.")

if not data_exists():
    st.warning("Os arquivos `outputs/X.csv` e `outputs/y.csv` ainda não existem.")
    if st.button("⚙️ Construir dataset de features agora"):
        build_data.clear()
        build_data()
        st.success("Pronto! Recarregue a página.")
    st.stop()

tabs = st.tabs(["📊 Visão geral", "✂️ Segmentação", "🏋️ Treinar & Comparar",
                "📈 Resultados por modelo", "🔍 XAI", "🔮 Predição"])

# =========================================================================== #
# 1. VISÃO GERAL                                                               #
# =========================================================================== #
with tabs[0]:
    st.subheader("Dataset")
    index = get_metadata()
    y = pd.read_csv(cfg.OUTPUTS / "y.csv")
    c1, c2, c3 = st.columns(3)
    c1.metric("Imagens (amostradas)", len(index))
    c2.metric("Classe fresh", int((y.label == 0).sum()))
    c3.metric("Classe rotten", int((y.label == 1).sum()))

    st.markdown("**Balanceamento por fruta e condição**")
    tab = index.groupby(["fruit", "label_name"]).size().unstack()
    st.dataframe(tab, use_container_width=True)
    st.bar_chart(tab)

    st.markdown("**Exemplos por classe**")
    if dataset_available():
        cols = st.columns(6)
        for col, folder in zip(cols, cfg.FOLDER_INFO):
            r = index[index.folder == folder].iloc[0]
            col.image(ds.load_image(r["path"]), caption=folder, use_container_width=True)
    else:
        fig_path = cfg.FIG_DIR / "exemplos_dataset.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Exemplos do dataset usados no projeto",
                     use_container_width=True)
        else:
            st.info("A pasta `dataset/` não está presente. Baixe o dataset para "
                    "exibir imagens individuais ou gere `outputs/figuras/exemplos_dataset.png`.")

    with st.expander("Por que usamos só as imagens originais?"):
        st.write(
            "O dataset do Kaggle traz, para cada foto, várias versões **aumentadas** "
            "(`rotated_by_*`, `translation_*`, `saltandpepper_*`, `vertical_flip_*`). "
            "Mantê-las causaria **vazamento de dados** (a mesma fruta em treino e "
            "teste) e prejudicaria a segmentação. Por isso usamos apenas as originais "
            "(`Screen Shot *`), amostradas de forma balanceada.")

# =========================================================================== #
# 2. SEGMENTAÇÃO                                                               #
# =========================================================================== #
with tabs[1]:
    st.subheader("Segmentação: Otsu × HSV")
    st.write("Compare os dois métodos. O **HSV** foi o escolhido para o pipeline.")
    index = get_metadata()
    colA, colB = st.columns([1, 2])
    with colA:
        choices = ["Enviar arquivo"]
        if dataset_available():
            choices.insert(0, "Do dataset")
        src_choice = st.radio("Imagem", choices)
        if src_choice == "Do dataset":
            folder = st.selectbox("Classe", list(cfg.FOLDER_INFO))
            sub = index[index.folder == folder].reset_index(drop=True)
            i = st.slider("Índice", 0, len(sub) - 1, 0)
            img = ds.load_image(sub.iloc[i]["path"])
        else:
            if not dataset_available():
                st.caption("A pasta `dataset/` não está presente; use uma imagem enviada "
                           "para testar a segmentação.")
            up = st.file_uploader("Imagem da fruta", type=["png", "jpg", "jpeg"],
                                  key="seg_up")
            img = read_image_file(up) if up else None

    with colB:
        if img is not None:
            m_otsu = seg.segment_otsu(img)
            m_hsv = seg.segment_hsv(img)
            g = st.columns(3)
            g[0].image(img, caption="Original", use_container_width=True)
            g[1].image(seg.apply_mask(img, m_otsu),
                       caption=f"Otsu · cobertura={seg.mask_coverage(m_otsu):.2f}",
                       use_container_width=True)
            g[2].image(seg.apply_mask(img, m_hsv),
                       caption=f"HSV · cobertura={seg.mask_coverage(m_hsv):.2f}",
                       use_container_width=True)
        else:
            st.info("Selecione ou envie uma imagem.")

# =========================================================================== #
# 3. TREINAR & COMPARAR                                                        #
# =========================================================================== #
with tabs[2]:
    st.subheader("Treinar e comparar os classificadores")
    st.write("Treina **KNN, Regressão Logística, Random Forest e SVM** com "
             "`GridSearchCV` (CV 5-fold) e avalia no conjunto de teste (split "
             "estratificado 60/20/20, sem vazamento).")

    cbtn = st.columns([1, 1, 3])
    if cbtn[0].button("🏋️ Treinar / comparar modelos", type="primary"):
        train_models.clear()
        st.session_state["res"] = train_models()
    if cbtn[1].button("🔄 Limpar cache de treino"):
        train_models.clear()
        st.session_state.pop("res", None)

    if "res" not in st.session_state and "res_autoload" not in st.session_state:
        # tenta treinar automaticamente na primeira visita
        st.session_state["res"] = train_models()
        st.session_state["res_autoload"] = True

    res = st.session_state.get("res")
    if res:
        table = res["table"].copy()
        best = table.iloc[0]["modelo"]
        st.success(f"Melhor modelo: **{best}**  (F1 = {table.iloc[0]['F1']:.3f}, "
                   f"ROC-AUC = {table.iloc[0]['ROC_AUC']:.3f})")

        st.markdown("**Tabela comparativa (conjunto de teste)**")
        st.dataframe(
            table.style.format({c: "{:.3f}" for c in
                                ["acuracia", "precisao", "recall", "F1", "ROC_AUC", "F1_cv"]})
            .background_gradient(cmap="Greens", subset=["F1", "ROC_AUC"]),
            use_container_width=True)

        # gráfico de barras das métricas
        st.markdown("**Métricas por modelo**")
        chart_df = table.set_index("modelo")[["acuracia", "precisao", "recall", "F1", "ROC_AUC"]]
        st.bar_chart(chart_df)

        # ROC de todos
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay
        st.markdown("**Curvas ROC**")
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, model in res["fitted"].items():
            RocCurveDisplay.from_estimator(model, res["split"].Xte, res["split"].yte,
                                           ax=ax, name=name)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.6)
        ax.set_title("ROC — conjunto de teste")
        st.pyplot(fig, use_container_width=False)

        with st.expander("Hiperparâmetros escolhidos (GridSearchCV)"):
            cv = pd.DataFrame([{"modelo": k, "F1_cv": v["F1_cv"], "params": str(v["params"])}
                               for k, v in res["cv_scores"].items()])
            st.dataframe(cv, use_container_width=True)

# =========================================================================== #
# 4. RESULTADOS POR MODELO                                                     #
# =========================================================================== #
with tabs[3]:
    st.subheader("Resultados detalhados por modelo")
    res = st.session_state.get("res")
    if not res:
        st.info("Treine os modelos na aba **Treinar & Comparar** primeiro.")
    else:
        name = st.selectbox("Modelo", list(res["fitted"]))
        model = res["fitted"][name]
        sp = res["split"]
        from sklearn.metrics import classification_report, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        yp = model.predict(sp.Xte)
        row = res["table"][res["table"].modelo == name].iloc[0]
        m = st.columns(5)
        m[0].metric("Acurácia", f"{row['acuracia']:.3f}")
        m[1].metric("Precisão", f"{row['precisao']:.3f}")
        m[2].metric("Recall", f"{row['recall']:.3f}")
        m[3].metric("F1", f"{row['F1']:.3f}")
        m[4].metric("ROC-AUC", f"{row['ROC_AUC']:.3f}")

        cc = st.columns(2)
        with cc[0]:
            st.markdown("**Matriz de confusão**")
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ConfusionMatrixDisplay(res["cms"][name], display_labels=["fresh", "rotten"]).plot(
                ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(name)
            st.pyplot(fig, use_container_width=False)
        with cc[1]:
            st.markdown("**Matriz normalizada (%)**")
            from sklearn.metrics import confusion_matrix
            cmn = confusion_matrix(sp.yte, yp, normalize="true")
            fig2, ax2 = plt.subplots(figsize=(4, 3.5))
            ConfusionMatrixDisplay(cmn, display_labels=["fresh", "rotten"]).plot(
                ax=ax2, cmap="Blues", colorbar=False, values_format=".2f")
            ax2.set_title(name)
            st.pyplot(fig2, use_container_width=False)

        st.markdown("**Relatório de classificação**")
        rep = classification_report(sp.yte, yp, target_names=["fresh", "rotten"],
                                    output_dict=True)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# =========================================================================== #
# 5. XAI                                                                       #
# =========================================================================== #
with tabs[4]:
    st.subheader("Explicabilidade (XAI)")
    res = st.session_state.get("res")
    if not res:
        st.info("Treine os modelos na aba **Treinar & Comparar** primeiro.")
    else:
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        feat = res["feat"]; sp = res["split"]
        rf = res["fitted"]["RandomForest"]; lg = res["fitted"]["LogReg"]

        x1, x2 = st.columns(2)
        with x1:
            st.markdown("**Random Forest — importância (Gini)**")
            imp = pd.Series(rf.named_steps["clf"].feature_importances_,
                            index=feat).sort_values(ascending=False).head(12)
            fig, ax = plt.subplots(figsize=(5, 4)); imp[::-1].plot.barh(ax=ax, color="#3c78b3")
            st.pyplot(fig, use_container_width=False)
        with x2:
            st.markdown("**Regressão Logística — coeficientes**")
            coef = pd.Series(lg.named_steps["clf"].coef_[0], index=feat).sort_values()
            top = pd.concat([coef.head(6), coef.tail(6)])
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh(top.index, top.values,
                    color=[GREEN if v < 0 else RED for v in top.values])
            ax.set_xlabel("coef (>0 → rotten)")
            st.pyplot(fig, use_container_width=False)

        if st.button("Calcular permutation importance"):
            with st.spinner("Calculando…"):
                pi = permutation_importance(rf, sp.Xte, sp.yte, scoring="f1",
                                            n_repeats=20, random_state=cfg.RANDOM_STATE,
                                            n_jobs=-1)
                pis = pd.Series(pi.importances_mean, index=feat).sort_values(ascending=False).head(12)
            fig, ax = plt.subplots(figsize=(6, 4)); pis[::-1].plot.barh(ax=ax, color="#8e44ad")
            ax.set_title("Permutation importance (queda de F1)")
            st.pyplot(fig, use_container_width=False)

        if st.button("Rodar ablation study por grupos"):
            with st.spinner("Treinando com cada grupo removido…"):
                X, yv = res["X"], res["y"]
                cvk = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)
                base = cross_val_score(RandomForestClassifier(300, random_state=cfg.RANDOM_STATE),
                                       X[feat], yv, cv=cvk, scoring="f1").mean()
                rows = []
                for g in ft.FEATURE_GROUPS:
                    cols = [c for c in feat if c not in ft.FEATURE_GROUPS[g]]
                    sc = cross_val_score(RandomForestClassifier(300, random_state=cfg.RANDOM_STATE),
                                         X[cols], yv, cv=cvk, scoring="f1").mean()
                    rows.append({"removido": g, "F1_cv": sc, "delta": sc - base})
                abl = pd.DataFrame(rows).sort_values("delta")
            st.dataframe(abl.round(4), use_container_width=True)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.barh(abl["removido"], abl["delta"],
                    color=[RED if v < 0 else GREEN for v in abl["delta"]])
            ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Δ F1 (negativo = grupo importante)")
            st.pyplot(fig, use_container_width=False)

        with st.expander("SHAP (pode demorar alguns segundos)"):
            if st.button("Calcular SHAP summary"):
                import shap
                with st.spinner("Calculando valores SHAP…"):
                    rf_clf = rf.named_steps["clf"]
                    Xte_sc = rf.named_steps["scaler"].transform(sp.Xte)
                    sv = shap.TreeExplainer(rf_clf).shap_values(Xte_sc)
                    sv_r = sv[..., 1] if (isinstance(sv, np.ndarray) and sv.ndim == 3) \
                        else (sv[1] if isinstance(sv, list) else sv)
                    fig = plt.figure()
                    shap.summary_plot(sv_r, sp.Xte, feature_names=feat, show=False, max_display=12)
                st.pyplot(fig, use_container_width=False)

        st.info("Os quatro métodos concordam: **cor/podridão** (`brown_ratio`, "
                "`mean_S`, `mean_V`) e **textura** (GLCM) dominam a decisão — coerente "
                "com o domínio e sem viés de fundo.")

# =========================================================================== #
# 6. PREDIÇÃO                                                                  #
# =========================================================================== #
with tabs[5]:
    st.subheader("Classificar uma fruta")
    res = st.session_state.get("res")
    if not res:
        st.info("Treine os modelos na aba **Treinar & Comparar** primeiro.")
    else:
        model_name = st.selectbox("Modelo para predição", list(res["fitted"]),
                                  index=list(res["fitted"]).index(res["table"].iloc[0]["modelo"]))
        up = st.file_uploader("Foto da fruta (fundo claro)", type=["png", "jpg", "jpeg"],
                              key="pred_up")
        if up:
            img = read_image_file(up)
            pred, proba, mask, feats, img_work = api.predict_image(
                img, res["fitted"][model_name], res["feat"])
            cc = st.columns(3)
            cc[0].image(img, caption="Original", use_container_width=True)
            cc[1].image(seg.apply_mask(img_work, mask), caption="Segmentação (HSV)",
                        use_container_width=True)
            with cc[2]:
                label = "🟥 ROTTEN (defeituosa)" if pred == 1 else "🟩 FRESH (OK)"
                st.markdown(f"### {label}")
                st.metric("Probabilidade de estar podre", f"{proba*100:.1f}%")
                st.progress(proba)

            st.markdown("**Por que? Comparação com as medianas de cada classe**")
            st.dataframe(api.feature_contributions(feats), use_container_width=True)
            st.caption("Valores próximos da coluna *mediana_rotten* puxam a decisão "
                       "para podre (ex.: `brown_ratio` e `dark_ratio` altos, `mean_S`/"
                       "`mean_V` baixos).")
        else:
            st.info("Envie uma imagem para classificar.")

st.divider()
st.caption("Pipeline clássico (segmentação HSV → features de forma/Hu/cor/textura → "
           "KNN/LogReg/RF/SVM) · `random_state=42` · scaler ajustado só no treino.")
