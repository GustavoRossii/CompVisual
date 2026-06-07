from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config as cfg
from . import dataset as ds
from . import features as ft
from . import modeling as mdl
from . import segmentation as seg

SEG_METHOD = "hsv"


# --------------------------------------------------------------------------- #
# Treino e avaliação de todos os modelos                                       #
# --------------------------------------------------------------------------- #
def train_all_models(progress=None):

    X, y, meta = mdl.load_Xy()
    sp = mdl.make_split(X, y)
    cvk = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)

    models = mdl.make_models()
    fitted, cv_scores, rows, cms = {}, {}, [], {}
    n = len(models)
    for i, (name, (pipe, grid)) in enumerate(models.items()):
        if progress:
            progress(i / n, f"Treinando {name}…")
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cvk, n_jobs=-1)
        gs.fit(sp.Xtrval, sp.ytrval)
        best = gs.best_estimator_
        fitted[name] = best
        cv_scores[name] = {"F1_cv": gs.best_score_, "params": gs.best_params_}

        yp = best.predict(sp.Xte)
        proba = best.predict_proba(sp.Xte)[:, 1]
        rows.append({
            "modelo": name,
            "acuracia": accuracy_score(sp.yte, yp),
            "precisao": precision_score(sp.yte, yp),
            "recall": recall_score(sp.yte, yp),
            "F1": f1_score(sp.yte, yp),
            "ROC_AUC": roc_auc_score(sp.yte, proba),
            "F1_cv": gs.best_score_,
        })
        cms[name] = confusion_matrix(sp.yte, yp)

    if progress:
        progress(1.0, "Concluído")

    table = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return {
        "fitted": fitted,
        "split": sp,
        "table": table,
        "cms": cms,
        "cv_scores": cv_scores,
        "meta": meta,
        "X": X, "y": y,
        "feat": ft.all_feature_names(),
    }


# --------------------------------------------------------------------------- #
# Predição de uma imagem nova                                                  #
# --------------------------------------------------------------------------- #
def predict_image(img_rgb: np.ndarray, model, feat_names: list[str]):
    img = _ensure_max_side(img_rgb, cfg.MAX_SIDE)
    mask = seg.segment(img, SEG_METHOD)
    feats = ft.extract_all(img, mask)
    x = pd.DataFrame([{k: feats[k] for k in feat_names}])
    pred = int(model.predict(x)[0])
    proba = float(model.predict_proba(x)[0, 1])
    return pred, proba, mask, feats, img


def _ensure_max_side(img_rgb: np.ndarray, max_side: int) -> np.ndarray:
    import cv2
    h, w = img_rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
    return img_rgb


def feature_contributions(feats: dict, top_k: int = 6) -> pd.DataFrame:
    """Compara as features de podridão/cor da imagem com a mediana de cada classe,
    para uma explicação simples no app."""
    X, y, _ = mdl.load_Xy()
    keys = ["brown_ratio", "dark_ratio", "mean_S", "mean_V",
            "glcm_homogeneity", "glcm_energy", "std_V", "sat_std"]
    fresh_med = X[y == 0][keys].median()
    rotten_med = X[y == 1][keys].median()
    rows = []
    for k in keys:
        rows.append({
            "feature": k,
            "valor_imagem": round(float(feats[k]), 3),
            "mediana_fresh": round(float(fresh_med[k]), 3),
            "mediana_rotten": round(float(rotten_med[k]), 3),
        })
    return pd.DataFrame(rows)
